/*
 * SPDX-FileCopyrightText: 2025 Wavira Project
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_ota_ops.h"
#include "esp_https_ota.h"
#include "esp_http_client.h"
#include "esp_app_format.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "cJSON.h"

#include "ota.h"

static const char *TAG = "wavira_ota";

/* OTA state */
static esp_mqtt_client_handle_t s_mqtt_client = NULL;
static TaskHandle_t s_ota_task_handle = NULL;
static SemaphoreHandle_t s_ota_mutex = NULL;
static ota_progress_t s_ota_progress = {0};
static char s_firmware_url[256] = {0};

/* Configuration */
#define OTA_RECV_TIMEOUT_MS     10000
#define OTA_BUFFER_SIZE         4096
#define OTA_PROGRESS_INTERVAL   5       /* Report progress every 5% */
#define OTA_TASK_STACK_SIZE     8192
#define OTA_TASK_PRIORITY       5

/* Forward declarations */
static void ota_task(void *pvParameter);
static void ota_report_progress(void);
static void ota_set_status(ota_status_t status, const char *error_msg);

/**
 * @brief HTTP event handler for OTA download progress tracking
 */
static esp_err_t ota_http_event_handler(esp_http_client_event_t *evt)
{
    switch (evt->event_id) {
        case HTTP_EVENT_ERROR:
            ESP_LOGD(TAG, "HTTP_EVENT_ERROR");
            break;
        case HTTP_EVENT_ON_CONNECTED:
            ESP_LOGD(TAG, "HTTP_EVENT_ON_CONNECTED");
            break;
        case HTTP_EVENT_ON_DATA:
            /* Track download progress */
            if (s_ota_mutex && xSemaphoreTake(s_ota_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                s_ota_progress.downloaded_size += evt->data_len;
                if (s_ota_progress.total_size > 0) {
                    uint8_t new_percent = (s_ota_progress.downloaded_size * 100) / s_ota_progress.total_size;
                    if (new_percent != s_ota_progress.progress_percent &&
                        (new_percent % OTA_PROGRESS_INTERVAL == 0 || new_percent == 100)) {
                        s_ota_progress.progress_percent = new_percent;
                        ota_report_progress();
                    }
                }
                xSemaphoreGive(s_ota_mutex);
            }
            break;
        case HTTP_EVENT_ON_FINISH:
            ESP_LOGD(TAG, "HTTP_EVENT_ON_FINISH");
            break;
        case HTTP_EVENT_DISCONNECTED:
            ESP_LOGD(TAG, "HTTP_EVENT_DISCONNECTED");
            break;
        default:
            break;
    }
    return ESP_OK;
}

/**
 * @brief Report OTA progress via MQTT
 */
static void ota_report_progress(void)
{
    if (!s_mqtt_client) return;

    char topic[128];
    snprintf(topic, sizeof(topic), "wavira/device/%s/ota/progress", CONFIG_WAVIRA_DEVICE_ID);

    cJSON *root = cJSON_CreateObject();
    if (!root) return;

    cJSON_AddStringToObject(root, "device_id", CONFIG_WAVIRA_DEVICE_ID);
    cJSON_AddNumberToObject(root, "timestamp", esp_timer_get_time() / 1000);

    const char *status_str;
    switch (s_ota_progress.status) {
        case OTA_STATUS_IDLE:
            status_str = "idle";
            break;
        case OTA_STATUS_DOWNLOADING:
            status_str = "downloading";
            break;
        case OTA_STATUS_VERIFYING:
            status_str = "verifying";
            break;
        case OTA_STATUS_APPLYING:
            status_str = "applying";
            break;
        case OTA_STATUS_SUCCESS:
            status_str = "success";
            break;
        case OTA_STATUS_FAILED:
            status_str = "failed";
            break;
        case OTA_STATUS_ROLLBACK:
            status_str = "rollback";
            break;
        default:
            status_str = "unknown";
            break;
    }
    cJSON_AddStringToObject(root, "status", status_str);
    cJSON_AddNumberToObject(root, "progress", s_ota_progress.progress_percent);
    cJSON_AddNumberToObject(root, "total_size", s_ota_progress.total_size);
    cJSON_AddNumberToObject(root, "downloaded_size", s_ota_progress.downloaded_size);

    if (s_ota_progress.error_msg[0] != '\0') {
        cJSON_AddStringToObject(root, "error", s_ota_progress.error_msg);
    }

    /* Add version info */
    const esp_app_desc_t *app_desc = esp_app_get_description();
    if (app_desc) {
        cJSON_AddStringToObject(root, "current_version", app_desc->version);
    }

    char *json_str = cJSON_PrintUnformatted(root);
    if (json_str) {
        esp_mqtt_client_publish(s_mqtt_client, topic, json_str, 0, 1, 0);
        ESP_LOGI(TAG, "OTA progress: %s (%d%%)", status_str, s_ota_progress.progress_percent);
        free(json_str);
    }
    cJSON_Delete(root);
}

/**
 * @brief Set OTA status with optional error message
 */
static void ota_set_status(ota_status_t status, const char *error_msg)
{
    if (s_ota_mutex && xSemaphoreTake(s_ota_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        s_ota_progress.status = status;
        if (error_msg) {
            strncpy(s_ota_progress.error_msg, error_msg, sizeof(s_ota_progress.error_msg) - 1);
            s_ota_progress.error_msg[sizeof(s_ota_progress.error_msg) - 1] = '\0';
        } else {
            s_ota_progress.error_msg[0] = '\0';
        }
        xSemaphoreGive(s_ota_mutex);
    }
    ota_report_progress();
}

/**
 * @brief OTA update task
 */
static void ota_task(void *pvParameter)
{
    ESP_LOGI(TAG, "Starting OTA from URL: %s", s_firmware_url);

    /* Reset progress */
    memset(&s_ota_progress, 0, sizeof(s_ota_progress));
    ota_set_status(OTA_STATUS_DOWNLOADING, NULL);

    /* Configure HTTP client */
    esp_http_client_config_t http_config = {
        .url = s_firmware_url,
        .timeout_ms = OTA_RECV_TIMEOUT_MS,
        .event_handler = ota_http_event_handler,
        .keep_alive_enable = true,
#ifdef CONFIG_WAVIRA_OTA_SKIP_CERT_VERIFY
        .skip_cert_common_name_check = true,
        .cert_pem = NULL,
#endif
    };

    /* Configure OTA */
    esp_https_ota_config_t ota_config = {
        .http_config = &http_config,
#ifdef CONFIG_WAVIRA_OTA_PARTIAL_HTTP
        .partial_http_download = true,
        .max_http_request_size = OTA_BUFFER_SIZE,
#endif
    };

    /* Get file size first via HEAD request */
    esp_http_client_handle_t client = esp_http_client_init(&http_config);
    if (client) {
        esp_http_client_set_method(client, HTTP_METHOD_HEAD);
        esp_err_t err = esp_http_client_perform(client);
        if (err == ESP_OK) {
            s_ota_progress.total_size = esp_http_client_get_content_length(client);
            ESP_LOGI(TAG, "Firmware size: %lu bytes", (unsigned long)s_ota_progress.total_size);
        }
        esp_http_client_cleanup(client);
    }

    /* Perform OTA update */
    esp_https_ota_handle_t ota_handle = NULL;
    esp_err_t err = esp_https_ota_begin(&ota_config, &ota_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "OTA begin failed: %s", esp_err_to_name(err));
        ota_set_status(OTA_STATUS_FAILED, esp_err_to_name(err));
        goto cleanup;
    }

    /* Get image info for size if HEAD request failed */
    if (s_ota_progress.total_size == 0) {
        s_ota_progress.total_size = esp_https_ota_get_image_size(ota_handle);
    }

    /* Download and write firmware */
    while (1) {
        err = esp_https_ota_perform(ota_handle);
        if (err != ESP_ERR_HTTPS_OTA_IN_PROGRESS) {
            break;
        }
        /* Task yield to allow other tasks to run */
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "OTA perform failed: %s", esp_err_to_name(err));
        ota_set_status(OTA_STATUS_FAILED, esp_err_to_name(err));
        esp_https_ota_abort(ota_handle);
        goto cleanup;
    }

    /* Verify firmware image */
    ota_set_status(OTA_STATUS_VERIFYING, NULL);

    if (!esp_https_ota_is_complete_data_received(ota_handle)) {
        ESP_LOGE(TAG, "Complete data not received");
        ota_set_status(OTA_STATUS_FAILED, "Incomplete download");
        esp_https_ota_abort(ota_handle);
        goto cleanup;
    }

    /* Apply the update */
    ota_set_status(OTA_STATUS_APPLYING, NULL);

    err = esp_https_ota_finish(ota_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "OTA finish failed: %s", esp_err_to_name(err));
        ota_set_status(OTA_STATUS_FAILED, esp_err_to_name(err));
        goto cleanup;
    }

    /* Success! */
    ota_set_status(OTA_STATUS_SUCCESS, NULL);
    ESP_LOGI(TAG, "OTA update successful! Rebooting in 3 seconds...");

    /* Delay before reboot to allow MQTT message to be sent */
    vTaskDelay(pdMS_TO_TICKS(3000));

    /* Reboot to apply new firmware */
    esp_restart();

cleanup:
    s_ota_task_handle = NULL;
    vTaskDelete(NULL);
}

/**
 * @brief Initialize OTA subsystem
 */
esp_err_t ota_init(esp_mqtt_client_handle_t mqtt_client)
{
    if (!mqtt_client) {
        ESP_LOGE(TAG, "Invalid MQTT client handle");
        return ESP_ERR_INVALID_ARG;
    }

    s_mqtt_client = mqtt_client;

    /* Create mutex for thread-safe access */
    s_ota_mutex = xSemaphoreCreateMutex();
    if (!s_ota_mutex) {
        ESP_LOGE(TAG, "Failed to create OTA mutex");
        return ESP_ERR_NO_MEM;
    }

    /* Check boot state and validate firmware if needed */
    const esp_partition_t *running = esp_ota_get_running_partition();
    esp_ota_img_states_t ota_state;

    if (esp_ota_get_state_partition(running, &ota_state) == ESP_OK) {
        if (ota_state == ESP_OTA_IMG_PENDING_VERIFY) {
            ESP_LOGI(TAG, "Firmware pending verification - marking as valid");
            /* Auto-validate after successful boot
             * In production, you might want manual validation after testing */
            esp_ota_mark_app_valid_cancel_rollback();
        }
    }

    /* Log current firmware info */
    const esp_app_desc_t *app_desc = esp_app_get_description();
    if (app_desc) {
        ESP_LOGI(TAG, "OTA initialized - Current firmware: %s", app_desc->version);
    }

    /* Check if rollback is available */
    if (ota_rollback_available()) {
        ESP_LOGI(TAG, "Previous firmware available for rollback");
    }

    return ESP_OK;
}

/**
 * @brief Start OTA update
 */
esp_err_t ota_start(const char *url)
{
    if (!url || strlen(url) == 0) {
        ESP_LOGE(TAG, "Invalid firmware URL");
        return ESP_ERR_INVALID_ARG;
    }

    if (s_ota_task_handle != NULL) {
        ESP_LOGW(TAG, "OTA already in progress");
        return ESP_ERR_INVALID_STATE;
    }

    /* Store URL */
    strncpy(s_firmware_url, url, sizeof(s_firmware_url) - 1);
    s_firmware_url[sizeof(s_firmware_url) - 1] = '\0';

    /* Create OTA task */
    BaseType_t ret = xTaskCreate(
        ota_task,
        "ota_task",
        OTA_TASK_STACK_SIZE,
        NULL,
        OTA_TASK_PRIORITY,
        &s_ota_task_handle
    );

    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create OTA task");
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "OTA update started");
    return ESP_OK;
}

/**
 * @brief Get current OTA progress
 */
esp_err_t ota_get_progress(ota_progress_t *progress)
{
    if (!progress) {
        return ESP_ERR_INVALID_ARG;
    }

    if (s_ota_mutex && xSemaphoreTake(s_ota_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        memcpy(progress, &s_ota_progress, sizeof(ota_progress_t));
        xSemaphoreGive(s_ota_mutex);
        return ESP_OK;
    }

    return ESP_ERR_TIMEOUT;
}

/**
 * @brief Mark current firmware as valid
 */
esp_err_t ota_mark_valid(void)
{
    esp_err_t err = esp_ota_mark_app_valid_cancel_rollback();
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Firmware marked as valid");
    } else {
        ESP_LOGE(TAG, "Failed to mark firmware valid: %s", esp_err_to_name(err));
    }
    return err;
}

/**
 * @brief Trigger rollback
 */
esp_err_t ota_rollback(void)
{
    ESP_LOGW(TAG, "Initiating rollback to previous firmware...");

    ota_set_status(OTA_STATUS_ROLLBACK, NULL);

    /* Delay to allow MQTT message */
    vTaskDelay(pdMS_TO_TICKS(1000));

    esp_err_t err = esp_ota_mark_app_invalid_rollback_and_reboot();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Rollback failed: %s", esp_err_to_name(err));
        ota_set_status(OTA_STATUS_FAILED, "Rollback failed");
    }
    return err;
}

/**
 * @brief Check if rollback is available
 */
bool ota_rollback_available(void)
{
    const esp_partition_t *running = esp_ota_get_running_partition();
    const esp_partition_t *other = esp_ota_get_next_update_partition(NULL);

    if (!running || !other || running == other) {
        return false;
    }

    esp_ota_img_states_t state;
    if (esp_ota_get_state_partition(other, &state) != ESP_OK) {
        return false;
    }

    /* Valid states for rollback */
    return (state == ESP_OTA_IMG_VALID ||
            state == ESP_OTA_IMG_UNDEFINED);
}

/**
 * @brief Get current firmware version
 */
const char *ota_get_version(void)
{
    const esp_app_desc_t *app_desc = esp_app_get_description();
    return app_desc ? app_desc->version : "unknown";
}

/**
 * @brief Handle MQTT OTA command
 */
void ota_handle_mqtt_command(const char *topic, const char *data, int data_len)
{
    if (!data || data_len == 0) {
        ESP_LOGW(TAG, "Empty OTA command");
        return;
    }

    /* Parse JSON command */
    cJSON *root = cJSON_ParseWithLength(data, data_len);
    if (!root) {
        ESP_LOGE(TAG, "Failed to parse OTA command JSON");
        return;
    }

    /* Get command type */
    cJSON *cmd = cJSON_GetObjectItem(root, "command");
    if (!cmd || !cJSON_IsString(cmd)) {
        ESP_LOGE(TAG, "Missing or invalid 'command' field");
        cJSON_Delete(root);
        return;
    }

    const char *cmd_str = cmd->valuestring;
    ESP_LOGI(TAG, "Received OTA command: %s", cmd_str);

    if (strcmp(cmd_str, "update") == 0) {
        /* Start OTA update */
        cJSON *url = cJSON_GetObjectItem(root, "url");
        if (url && cJSON_IsString(url)) {
            ota_start(url->valuestring);
        } else {
            ESP_LOGE(TAG, "Missing firmware URL in update command");
            ota_set_status(OTA_STATUS_FAILED, "Missing URL");
        }
    } else if (strcmp(cmd_str, "rollback") == 0) {
        /* Trigger rollback */
        if (ota_rollback_available()) {
            ota_rollback();
        } else {
            ESP_LOGW(TAG, "Rollback not available");
            ota_set_status(OTA_STATUS_FAILED, "No rollback available");
        }
    } else if (strcmp(cmd_str, "status") == 0) {
        /* Report current status */
        ota_report_progress();
    } else if (strcmp(cmd_str, "version") == 0) {
        /* Report version info */
        char version_topic[128];
        snprintf(version_topic, sizeof(version_topic),
                 "wavira/device/%s/version", CONFIG_WAVIRA_DEVICE_ID);

        const esp_app_desc_t *app_desc = esp_app_get_description();
        if (app_desc) {
            cJSON *ver = cJSON_CreateObject();
            cJSON_AddStringToObject(ver, "device_id", CONFIG_WAVIRA_DEVICE_ID);
            cJSON_AddStringToObject(ver, "version", app_desc->version);
            cJSON_AddStringToObject(ver, "idf_version", app_desc->idf_ver);
            cJSON_AddStringToObject(ver, "project_name", app_desc->project_name);
            cJSON_AddStringToObject(ver, "compile_date", app_desc->date);
            cJSON_AddStringToObject(ver, "compile_time", app_desc->time);
            cJSON_AddBoolToObject(ver, "rollback_available", ota_rollback_available());

            char *ver_str = cJSON_PrintUnformatted(ver);
            if (ver_str) {
                esp_mqtt_client_publish(s_mqtt_client, version_topic, ver_str, 0, 1, 0);
                free(ver_str);
            }
            cJSON_Delete(ver);
        }
    } else {
        ESP_LOGW(TAG, "Unknown OTA command: %s", cmd_str);
    }

    cJSON_Delete(root);
}
