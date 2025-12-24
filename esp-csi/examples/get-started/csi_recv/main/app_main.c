/*
 * SPDX-FileCopyrightText: 2025 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file app_main.c
 * @brief ESP32 CSI Receiver with HTTP/MQTT/Wi-Fi Transmission Support
 *
 * This firmware supports three output modes:
 * - Serial Mode: Traditional UART output for development/debugging
 * - HTTP Mode: Wi-Fi transmission to a server via HTTP
 * - MQTT Mode: Wi-Fi transmission to a broker via MQTT (recommended)
 *
 * Issue #15: ESP32ファームウェアにWi-Fi経由HTTP送信機能を追加
 * Issue #21-26: MQTT対応追加
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"

#include "nvs_flash.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_event.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "driver/gpio.h"

#ifdef CONFIG_CSI_OUTPUT_HTTP
#include "esp_http_client.h"
#include "cJSON.h"
#include "lwip/inet.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "ping/ping_sock.h"
#endif

#ifdef CONFIG_CSI_OUTPUT_MQTT
#include "mqtt_client.h"
#include "cJSON.h"
#endif

static const char *TAG = "wavira_csi";

// =============================================================================
// Configuration Defaults (can be overridden by Kconfig)
// =============================================================================

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
#define CONFIG_WIFI_BAND_MODE   WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT40
#define CONFIG_WIFI_5G_BANDWIDTHS           WIFI_BW_HT40
#define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11N
#define CONFIG_WIFI_5G_PROTOCOL             WIFI_PROTOCOL_11N
#define CONFIG_ESP_NOW_PHYMODE           WIFI_PHY_MODE_HT40
#else
#define CONFIG_WIFI_BANDWIDTH           WIFI_BW_HT40
#endif

#define CONFIG_ESP_NOW_RATE             WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN                   1

#if CONFIG_IDF_TARGET_ESP32C5
#define CSI_FORCE_LLTF                      0
#endif

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
#define CONFIG_GAIN_CONTROL                 1
#endif

// Default values if not defined in Kconfig
#ifndef CONFIG_WAVIRA_BUFFER_SIZE
#define CONFIG_WAVIRA_BUFFER_SIZE 50
#endif

#ifndef CONFIG_WAVIRA_BATCH_SIZE
#define CONFIG_WAVIRA_BATCH_SIZE 10
#endif

#ifndef CONFIG_WAVIRA_SEND_INTERVAL_MS
#define CONFIG_WAVIRA_SEND_INTERVAL_MS 100
#endif

#ifndef CONFIG_WAVIRA_HTTP_TIMEOUT_MS
#define CONFIG_WAVIRA_HTTP_TIMEOUT_MS 5000
#endif

#ifndef CONFIG_WAVIRA_HTTP_MAX_RETRIES
#define CONFIG_WAVIRA_HTTP_MAX_RETRIES 3
#endif

#ifndef CONFIG_WAVIRA_LED_GPIO
#define CONFIG_WAVIRA_LED_GPIO 2
#endif

#ifndef CONFIG_CSI_TRIGGER_FREQUENCY
#define CONFIG_CSI_TRIGGER_FREQUENCY 10
#endif

// =============================================================================
// PHY Structure for Gain Control
// =============================================================================

typedef struct {
    unsigned : 32;
    unsigned : 32;
    unsigned : 32;
    unsigned : 32;
    unsigned : 32;
#if CONFIG_IDF_TARGET_ESP32S2
    unsigned : 32;
#elif CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
    unsigned : 16;
    unsigned fft_gain : 8;
    unsigned agc_gain : 8;
    unsigned : 32;
#endif
    unsigned : 32;
#if CONFIG_IDF_TARGET_ESP32S2
    signed : 8;
    unsigned : 24;
#elif CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5
    unsigned : 32;
    unsigned : 32;
    unsigned : 32;
#endif
    unsigned : 32;
} wifi_pkt_rx_ctrl_phy_t;

#if CONFIG_FORCE_GAIN
extern void phy_fft_scale_force(bool force_en, uint8_t force_value);
extern void phy_force_rx_gain(int force_en, int force_value);
#endif

// =============================================================================
// CSI Data Structure for Buffering
// =============================================================================

typedef struct {
    uint32_t seq;
    uint8_t mac[6];
    int8_t rssi;
    uint8_t rate;
    uint8_t channel;
    uint32_t timestamp;
    int64_t local_timestamp_ms;
    uint16_t sig_len;
    uint8_t rx_state;
    uint8_t fft_gain;
    uint8_t agc_gain;
    int16_t noise_floor;
    uint16_t csi_len;
    int8_t csi_data[640];  // Max CSI data length (increased for HT40/multiple antennas)
} csi_packet_t;

// =============================================================================
// Global Variables
// =============================================================================

#ifdef CONFIG_CSI_TRIGGER_ESPNOW
static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
#endif

#if defined(CONFIG_CSI_OUTPUT_HTTP) || defined(CONFIG_CSI_OUTPUT_MQTT)
static EventGroupHandle_t s_wifi_event_group;
static QueueHandle_t s_csi_queue;
static volatile bool s_wifi_connected = false;
static uint32_t s_total_packets_sent = 0;
static uint32_t s_total_packets_failed = 0;
static int64_t s_boot_time_ms = 0;

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1
#endif

#ifdef CONFIG_CSI_OUTPUT_MQTT
static esp_mqtt_client_handle_t s_mqtt_client = NULL;
static volatile bool s_mqtt_connected = false;

// MQTT Topic for CSI data batch
static char s_mqtt_csi_topic[128];
static char s_mqtt_status_topic[128];
static char s_mqtt_will_topic[128];
#endif

static uint32_t s_packet_count = 0;

// =============================================================================
// LED Status Indicator
// =============================================================================

typedef enum {
    LED_OFF = 0,
    LED_WIFI_CONNECTING,
    LED_WIFI_CONNECTED,
    LED_TRANSMITTING,
    LED_ERROR
} led_state_t;

static volatile led_state_t s_led_state = LED_OFF;

static void led_task(void *pvParameter)
{
    gpio_reset_pin(CONFIG_WAVIRA_LED_GPIO);
    gpio_set_direction(CONFIG_WAVIRA_LED_GPIO, GPIO_MODE_OUTPUT);

    while (1) {
        switch (s_led_state) {
            case LED_OFF:
                gpio_set_level(CONFIG_WAVIRA_LED_GPIO, 0);
                vTaskDelay(500 / portTICK_PERIOD_MS);
                break;
            case LED_WIFI_CONNECTING:
                gpio_set_level(CONFIG_WAVIRA_LED_GPIO, 1);
                vTaskDelay(100 / portTICK_PERIOD_MS);
                gpio_set_level(CONFIG_WAVIRA_LED_GPIO, 0);
                vTaskDelay(100 / portTICK_PERIOD_MS);
                break;
            case LED_WIFI_CONNECTED:
                gpio_set_level(CONFIG_WAVIRA_LED_GPIO, 1);
                vTaskDelay(1000 / portTICK_PERIOD_MS);
                break;
            case LED_TRANSMITTING:
                gpio_set_level(CONFIG_WAVIRA_LED_GPIO, 1);
                vTaskDelay(50 / portTICK_PERIOD_MS);
                gpio_set_level(CONFIG_WAVIRA_LED_GPIO, 0);
                vTaskDelay(50 / portTICK_PERIOD_MS);
                break;
            case LED_ERROR:
                gpio_set_level(CONFIG_WAVIRA_LED_GPIO, 1);
                vTaskDelay(200 / portTICK_PERIOD_MS);
                gpio_set_level(CONFIG_WAVIRA_LED_GPIO, 0);
                vTaskDelay(200 / portTICK_PERIOD_MS);
                gpio_set_level(CONFIG_WAVIRA_LED_GPIO, 1);
                vTaskDelay(200 / portTICK_PERIOD_MS);
                gpio_set_level(CONFIG_WAVIRA_LED_GPIO, 0);
                vTaskDelay(600 / portTICK_PERIOD_MS);
                break;
        }
    }
}

// =============================================================================
// HTTP Mode Implementation
// =============================================================================

#if defined(CONFIG_CSI_OUTPUT_HTTP) || defined(CONFIG_CSI_OUTPUT_MQTT)

static int s_wifi_retry_count = 0;
static esp_timer_handle_t s_reconnect_timer = NULL;

/**
 * @brief Timer callback for Wi-Fi reconnection with exponential backoff
 * @note Using timer instead of vTaskDelay to avoid blocking the event loop
 */
static void wifi_reconnect_timer_cb(void *arg)
{
    ESP_LOGI(TAG, "Attempting Wi-Fi reconnection (attempt %d)...", s_wifi_retry_count + 1);
    esp_wifi_connect();
    s_wifi_retry_count++;
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_wifi_connected = false;
        s_led_state = LED_WIFI_CONNECTING;

        if (CONFIG_WAVIRA_WIFI_MAXIMUM_RETRY == 0 ||
            s_wifi_retry_count < CONFIG_WAVIRA_WIFI_MAXIMUM_RETRY) {
            // Exponential backoff using timer (non-blocking)
            uint64_t delay_ms = (1ULL << (s_wifi_retry_count > 5 ? 5 : s_wifi_retry_count)) * 100;
            ESP_LOGI(TAG, "Wi-Fi disconnected, will retry in %llu ms (attempt %d)...",
                     delay_ms, s_wifi_retry_count + 1);

            // Schedule reconnection via timer to avoid blocking event loop
            if (s_reconnect_timer) {
                esp_timer_stop(s_reconnect_timer);
                esp_timer_start_once(s_reconnect_timer, delay_ms * 1000);  // Convert to microseconds
            }
        } else {
            ESP_LOGE(TAG, "Wi-Fi connection failed after %d attempts", s_wifi_retry_count);
            s_led_state = LED_ERROR;
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_wifi_retry_count = 0;
        s_wifi_connected = true;
        s_led_state = LED_WIFI_CONNECTED;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    // Create reconnection timer for non-blocking exponential backoff
    const esp_timer_create_args_t reconnect_timer_args = {
        .callback = &wifi_reconnect_timer_cb,
        .name = "wifi_reconnect"
    };
    ESP_ERROR_CHECK(esp_timer_create(&reconnect_timer_args, &s_reconnect_timer));

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                    ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                    IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = CONFIG_WAVIRA_WIFI_SSID,
            .password = CONFIG_WAVIRA_WIFI_PASSWORD,
            .threshold.authmode = CONFIG_WAVIRA_WIFI_SCAN_AUTH_MODE_THRESHOLD,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi STA initialization complete, connecting to %s...",
             CONFIG_WAVIRA_WIFI_SSID);
    s_led_state = LED_WIFI_CONNECTING;

    // Wait for connection
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                           pdFALSE, pdFALSE, portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to Wi-Fi SSID: %s", CONFIG_WAVIRA_WIFI_SSID);
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGE(TAG, "Failed to connect to Wi-Fi SSID: %s", CONFIG_WAVIRA_WIFI_SSID);
    }
}

static cJSON *csi_packet_to_json(csi_packet_t *pkt)
{
    cJSON *json = cJSON_CreateObject();
    if (!json) return NULL;

    cJSON_AddStringToObject(json, "device_id", CONFIG_WAVIRA_DEVICE_ID);
    // TODO: For absolute timestamps, implement SNTP synchronization:
    //   1. Call esp_sntp_init() and esp_sntp_setservername() at startup
    //   2. Wait for SNTP sync: sntp_get_sync_status() == SNTP_SYNC_STATUS_COMPLETED
    //   3. Use time() to get Unix timestamp instead of esp_timer_get_time()
    // Current timestamp is milliseconds since boot (relative time)
    cJSON_AddNumberToObject(json, "timestamp", pkt->local_timestamp_ms);
    cJSON_AddNumberToObject(json, "seq", pkt->seq);

    char mac_str[18];
    snprintf(mac_str, sizeof(mac_str), "%02x:%02x:%02x:%02x:%02x:%02x",
             pkt->mac[0], pkt->mac[1], pkt->mac[2],
             pkt->mac[3], pkt->mac[4], pkt->mac[5]);
    cJSON_AddStringToObject(json, "mac", mac_str);

    cJSON_AddNumberToObject(json, "rssi", pkt->rssi);
    cJSON_AddNumberToObject(json, "rate", pkt->rate);
    cJSON_AddNumberToObject(json, "channel", pkt->channel);

    // Convert CSI I/Q data to amplitudes
    cJSON *csi_array = cJSON_CreateArray();
    if (csi_array) {
        for (int i = 0; i < pkt->csi_len - 1; i += 2) {
            float I = (float)pkt->csi_data[i];
            float Q = (float)pkt->csi_data[i + 1];
            float amplitude = sqrtf(I * I + Q * Q);
            cJSON_AddItemToArray(csi_array, cJSON_CreateNumber(amplitude));
        }
        cJSON_AddItemToObject(json, "csi_data", csi_array);
    }

    // Metadata
    cJSON *metadata = cJSON_CreateObject();
    if (metadata) {
        cJSON_AddStringToObject(metadata, "firmware_version", "1.0.0");
        cJSON_AddNumberToObject(metadata, "uptime_ms",
                                (esp_timer_get_time() / 1000) - s_boot_time_ms);
        cJSON_AddItemToObject(json, "metadata", metadata);
    }

    return json;
}

static esp_err_t http_send_csi_batch(csi_packet_t *packets, int count)
{
    if (!s_wifi_connected || count == 0) {
        return ESP_ERR_INVALID_STATE;
    }

    s_led_state = LED_TRANSMITTING;

    // Create batch JSON
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        ESP_LOGE(TAG, "Failed to create JSON object");
        return ESP_ERR_NO_MEM;
    }

    cJSON_AddStringToObject(root, "device_id", CONFIG_WAVIRA_DEVICE_ID);

    cJSON *data_array = cJSON_CreateArray();
    if (!data_array) {
        cJSON_Delete(root);
        return ESP_ERR_NO_MEM;
    }

    for (int i = 0; i < count; i++) {
        cJSON *pkt_json = csi_packet_to_json(&packets[i]);
        if (pkt_json) {
            cJSON_AddItemToArray(data_array, pkt_json);
        }
    }

    cJSON_AddItemToObject(root, "data", data_array);

    char *json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json_str) {
        ESP_LOGE(TAG, "Failed to serialize JSON");
        return ESP_ERR_NO_MEM;
    }

    // HTTP client configuration
    // TODO: For production deployment, enable HTTPS/TLS support:
    //   1. Add CONFIG_WAVIRA_USE_HTTPS option in Kconfig
    //   2. Set .transport_type = HTTP_TRANSPORT_OVER_SSL
    //   3. Set .cert_pem = server_cert_pem (from embed file or config)
    //   4. Consider using ESP-IDF certificate bundle: esp_crt_bundle_attach
    esp_http_client_config_t config = {
        .url = CONFIG_WAVIRA_BATCH_ENDPOINT,
        .method = HTTP_METHOD_POST,
        .timeout_ms = CONFIG_WAVIRA_HTTP_TIMEOUT_MS,
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) {
        free(json_str);
        return ESP_ERR_NO_MEM;
    }

    esp_http_client_set_header(client, "Content-Type", "application/json");

#ifdef CONFIG_WAVIRA_API_KEY
    if (strlen(CONFIG_WAVIRA_API_KEY) > 0) {
        esp_http_client_set_header(client, "X-API-Key", CONFIG_WAVIRA_API_KEY);
    }
#endif

    esp_http_client_set_post_field(client, json_str, strlen(json_str));

    esp_err_t err = ESP_FAIL;
    int retry_count = 0;

    while (retry_count <= CONFIG_WAVIRA_HTTP_MAX_RETRIES) {
        err = esp_http_client_perform(client);

        if (err == ESP_OK) {
            int status = esp_http_client_get_status_code(client);
            if (status == 200) {
                s_total_packets_sent += count;
#ifdef CONFIG_WAVIRA_DEBUG_SERIAL
                ESP_LOGI(TAG, "Sent %d packets (total: %lu)", count, s_total_packets_sent);
#endif
                break;
            } else {
                ESP_LOGW(TAG, "HTTP status: %d", status);
                err = ESP_FAIL;
            }
        }

        retry_count++;
        if (retry_count <= CONFIG_WAVIRA_HTTP_MAX_RETRIES) {
            ESP_LOGW(TAG, "HTTP retry %d/%d", retry_count, CONFIG_WAVIRA_HTTP_MAX_RETRIES);
            vTaskDelay((100 * retry_count) / portTICK_PERIOD_MS);
        }
    }

    if (err != ESP_OK) {
        s_total_packets_failed += count;
        s_led_state = LED_ERROR;
        ESP_LOGE(TAG, "HTTP send failed: %s", esp_err_to_name(err));
    } else {
        s_led_state = LED_WIFI_CONNECTED;
    }

    esp_http_client_cleanup(client);
    free(json_str);

    return err;
}

static void http_sender_task(void *pvParameter)
{
    csi_packet_t batch[CONFIG_WAVIRA_BATCH_SIZE];
    int batch_count = 0;
    int64_t last_send_time = esp_timer_get_time() / 1000;

    ESP_LOGI(TAG, "HTTP sender task started (batch_size=%d, interval=%dms)",
             CONFIG_WAVIRA_BATCH_SIZE, CONFIG_WAVIRA_SEND_INTERVAL_MS);

    while (1) {
        csi_packet_t pkt;

        // Try to receive from queue with timeout
        if (xQueueReceive(s_csi_queue, &pkt, pdMS_TO_TICKS(CONFIG_WAVIRA_SEND_INTERVAL_MS))) {
            batch[batch_count++] = pkt;
        }

        int64_t now = esp_timer_get_time() / 1000;
        bool should_send = false;

        // Send if batch is full or interval elapsed with data
        if (batch_count >= CONFIG_WAVIRA_BATCH_SIZE) {
            should_send = true;
        } else if (batch_count > 0 && (now - last_send_time) >= CONFIG_WAVIRA_SEND_INTERVAL_MS) {
            should_send = true;
        }

        if (should_send && s_wifi_connected) {
            http_send_csi_batch(batch, batch_count);
            batch_count = 0;
            last_send_time = now;
        }
    }
}

#ifdef CONFIG_CSI_TRIGGER_ROUTER
static void wifi_ping_router_start(void)
{
    wifi_ap_record_t ap_info;
    ESP_ERROR_CHECK(esp_wifi_sta_get_ap_info(&ap_info));

    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
    ping_config.count = 0;  // Infinite
    ping_config.interval_ms = 1000 / CONFIG_CSI_TRIGGER_FREQUENCY;
    ping_config.task_stack_size = 3072;
    ping_config.data_size = 1;

    // Get gateway IP
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    esp_netif_ip_info_t ip_info;
    esp_netif_get_ip_info(netif, &ip_info);

    ping_config.target_addr.u_addr.ip4.addr = ip_info.gw.addr;
    ping_config.target_addr.type = ESP_IPADDR_TYPE_V4;

    esp_ping_handle_t ping_handle;
    esp_ping_callbacks_t cbs = {0};
    ESP_ERROR_CHECK(esp_ping_new_session(&ping_config, &cbs, &ping_handle));
    ESP_ERROR_CHECK(esp_ping_start(ping_handle));

    ESP_LOGI(TAG, "Started pinging gateway at %d Hz", CONFIG_CSI_TRIGGER_FREQUENCY);
}
#endif

#endif // CONFIG_CSI_OUTPUT_HTTP

// =============================================================================
// MQTT Functions
// =============================================================================

#ifdef CONFIG_CSI_OUTPUT_MQTT

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    esp_mqtt_event_handle_t event = event_data;

    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT Connected to broker");
            s_mqtt_connected = true;
            s_led_state = LED_WIFI_CONNECTED;

            // Publish online status
            char status_msg[128];
            snprintf(status_msg, sizeof(status_msg),
                     "{\"device_id\":\"%s\",\"status\":\"online\",\"timestamp\":%lld}",
                     CONFIG_WAVIRA_DEVICE_ID, esp_timer_get_time() / 1000);
            esp_mqtt_client_publish(s_mqtt_client, s_mqtt_status_topic, status_msg, 0,
                                    CONFIG_WAVIRA_MQTT_QOS, 1);  // retained
            break;

        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGW(TAG, "MQTT Disconnected from broker");
            s_mqtt_connected = false;
            s_led_state = LED_WIFI_CONNECTING;
            break;

        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT Error: type=%d", event->error_handle->error_type);
            if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                ESP_LOGE(TAG, "Last error code: 0x%x", event->error_handle->esp_tls_last_esp_err);
            }
            break;

        case MQTT_EVENT_PUBLISHED:
            ESP_LOGD(TAG, "MQTT Message published, msg_id=%d", event->msg_id);
            break;

        default:
            ESP_LOGD(TAG, "MQTT Event: %d", event_id);
            break;
    }
}

static esp_err_t mqtt_publish_csi_batch(csi_packet_t *packets, int count)
{
    if (!s_mqtt_connected || !s_mqtt_client || count == 0) {
        return ESP_ERR_INVALID_STATE;
    }

    s_led_state = LED_TRANSMITTING;

    // Create batch JSON
    cJSON *root = cJSON_CreateObject();
    if (!root) {
        ESP_LOGE(TAG, "Failed to create JSON object");
        return ESP_ERR_NO_MEM;
    }

    cJSON_AddStringToObject(root, "device_id", CONFIG_WAVIRA_DEVICE_ID);
    cJSON_AddNumberToObject(root, "timestamp", esp_timer_get_time() / 1000);

    cJSON *batch_array = cJSON_CreateArray();
    if (!batch_array) {
        cJSON_Delete(root);
        return ESP_ERR_NO_MEM;
    }

    for (int i = 0; i < count; i++) {
        cJSON *packet_json = create_csi_json(&packets[i]);
        if (packet_json) {
            cJSON_AddItemToArray(batch_array, packet_json);
        }
    }

    cJSON_AddItemToObject(root, "batch", batch_array);

    char *json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json_str) {
        ESP_LOGE(TAG, "Failed to print JSON");
        return ESP_ERR_NO_MEM;
    }

    int msg_id = esp_mqtt_client_publish(s_mqtt_client, s_mqtt_csi_topic,
                                          json_str, 0, CONFIG_WAVIRA_MQTT_QOS, 0);
    free(json_str);

    if (msg_id < 0) {
        ESP_LOGE(TAG, "MQTT publish failed");
        s_total_packets_failed += count;
        s_led_state = LED_ERROR;
        return ESP_FAIL;
    }

    s_total_packets_sent += count;
    s_led_state = LED_WIFI_CONNECTED;

    ESP_LOGD(TAG, "Published %d CSI packets, msg_id=%d", count, msg_id);
    return ESP_OK;
}

static void mqtt_sender_task(void *pvParameter)
{
    csi_packet_t batch_buffer[CONFIG_WAVIRA_BATCH_SIZE];
    int batch_count = 0;
    TickType_t last_send_time = xTaskGetTickCount();

    ESP_LOGI(TAG, "MQTT sender task started");

    while (1) {
        csi_packet_t packet;
        TickType_t wait_time = pdMS_TO_TICKS(CONFIG_WAVIRA_SEND_INTERVAL_MS);

        // Try to receive a packet from queue (with timeout)
        if (xQueueReceive(s_csi_queue, &packet, wait_time) == pdTRUE) {
            // Add packet to batch buffer
            if (batch_count < CONFIG_WAVIRA_BATCH_SIZE) {
                memcpy(&batch_buffer[batch_count], &packet, sizeof(csi_packet_t));
                batch_count++;
            }
        }

        // Check if we should send the batch
        TickType_t current_time = xTaskGetTickCount();
        bool time_elapsed = (current_time - last_send_time) >= pdMS_TO_TICKS(CONFIG_WAVIRA_SEND_INTERVAL_MS);
        bool buffer_full = (batch_count >= CONFIG_WAVIRA_BATCH_SIZE);

        if ((time_elapsed || buffer_full) && batch_count > 0 && s_mqtt_connected) {
            esp_err_t err = mqtt_publish_csi_batch(batch_buffer, batch_count);
            if (err == ESP_OK) {
                ESP_LOGD(TAG, "Batch sent: %d packets", batch_count);
            } else if (err != ESP_ERR_INVALID_STATE) {
                ESP_LOGW(TAG, "Batch send failed: %s", esp_err_to_name(err));
            }
            batch_count = 0;
            last_send_time = current_time;
        }

        // Small yield to prevent watchdog
        taskYIELD();
    }
}

static void mqtt_init(void)
{
    // Build topic strings
    snprintf(s_mqtt_csi_topic, sizeof(s_mqtt_csi_topic),
             "wavira/device/%s/csi/batch", CONFIG_WAVIRA_DEVICE_ID);
    snprintf(s_mqtt_status_topic, sizeof(s_mqtt_status_topic),
             "wavira/device/%s/status", CONFIG_WAVIRA_DEVICE_ID);
    snprintf(s_mqtt_will_topic, sizeof(s_mqtt_will_topic),
             "wavira/device/%s/will", CONFIG_WAVIRA_DEVICE_ID);

    // Build Last Will message
    char will_msg[128];
    snprintf(will_msg, sizeof(will_msg),
             "{\"device_id\":\"%s\",\"status\":\"offline\",\"reason\":\"unexpected_disconnect\"}",
             CONFIG_WAVIRA_DEVICE_ID);

    // Configure MQTT client
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker = {
            .address = {
                .uri = CONFIG_WAVIRA_MQTT_BROKER_URL,
                .port = CONFIG_WAVIRA_MQTT_PORT,
            },
        },
        .credentials = {
            .username = strlen(CONFIG_WAVIRA_MQTT_USERNAME) > 0 ? CONFIG_WAVIRA_MQTT_USERNAME : NULL,
            .authentication = {
                .password = strlen(CONFIG_WAVIRA_MQTT_PASSWORD) > 0 ? CONFIG_WAVIRA_MQTT_PASSWORD : NULL,
            },
            .client_id = CONFIG_WAVIRA_MQTT_CLIENT_ID,
        },
        .session = {
            .keepalive = CONFIG_WAVIRA_MQTT_KEEPALIVE,
            .last_will = {
                .topic = s_mqtt_will_topic,
                .msg = will_msg,
                .msg_len = strlen(will_msg),
                .qos = CONFIG_WAVIRA_MQTT_QOS,
                .retain = true,
            },
        },
    };

    ESP_LOGI(TAG, "Initializing MQTT client...");
    ESP_LOGI(TAG, "  Broker: %s:%d", CONFIG_WAVIRA_MQTT_BROKER_URL, CONFIG_WAVIRA_MQTT_PORT);
    ESP_LOGI(TAG, "  Client ID: %s", CONFIG_WAVIRA_MQTT_CLIENT_ID);
    ESP_LOGI(TAG, "  CSI Topic: %s", s_mqtt_csi_topic);

    s_mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    if (!s_mqtt_client) {
        ESP_LOGE(TAG, "Failed to initialize MQTT client");
        return;
    }

    esp_mqtt_client_register_event(s_mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(s_mqtt_client);

    ESP_LOGI(TAG, "MQTT client started");
}

#endif // CONFIG_CSI_OUTPUT_MQTT

// =============================================================================
// CSI Callback (Common for all modes)
// =============================================================================

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        ESP_LOGW(TAG, "Invalid CSI callback arguments");
        return;
    }

    // Filter by sender MAC (for ESP-NOW mode)
#ifdef CONFIG_CSI_TRIGGER_ESPNOW
    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) {
        return;
    }
#endif

    wifi_pkt_rx_ctrl_phy_t *phy_info = (wifi_pkt_rx_ctrl_phy_t *)info;
    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;

    // Gain control (first 100 packets)
#if CONFIG_GAIN_CONTROL
    static uint16_t agc_gain_sum = 0;
    static uint16_t fft_gain_sum = 0;
    static uint8_t agc_gain_force_value = 0;
    static uint8_t fft_gain_force_value = 0;

    if (s_packet_count < 100) {
        agc_gain_sum += phy_info->agc_gain;
        fft_gain_sum += phy_info->fft_gain;
    } else if (s_packet_count == 100) {
        agc_gain_force_value = agc_gain_sum / 100;
        fft_gain_force_value = fft_gain_sum / 100;
#if CONFIG_FORCE_GAIN
        phy_fft_scale_force(1, fft_gain_force_value);
        phy_force_rx_gain(1, agc_gain_force_value);
#endif
        ESP_LOGI(TAG, "Gain calibrated: fft=%d, agc=%d", fft_gain_force_value, agc_gain_force_value);
    }
#endif

#if defined(CONFIG_CSI_OUTPUT_HTTP) || defined(CONFIG_CSI_OUTPUT_MQTT)
    // HTTP/MQTT Mode: Queue the packet for transmission
#ifdef CONFIG_CSI_OUTPUT_MQTT
    bool is_connected = s_wifi_connected && s_mqtt_connected;
#else
    bool is_connected = s_wifi_connected;
#endif
    if (is_connected && s_csi_queue) {
        csi_packet_t pkt = {0};

#ifdef CONFIG_CSI_TRIGGER_ESPNOW
        pkt.seq = *(uint32_t *)(info->payload + 15);
#else
        pkt.seq = s_packet_count;
#endif

        memcpy(pkt.mac, info->mac, 6);
        pkt.rssi = rx_ctrl->rssi;
        pkt.rate = rx_ctrl->rate;
        pkt.channel = rx_ctrl->channel;
        pkt.timestamp = rx_ctrl->timestamp;
        pkt.local_timestamp_ms = esp_timer_get_time() / 1000;
        pkt.sig_len = rx_ctrl->sig_len;
        pkt.rx_state = rx_ctrl->rx_state;

#if CONFIG_GAIN_CONTROL
        pkt.fft_gain = phy_info->fft_gain;
        pkt.agc_gain = phy_info->agc_gain;
#endif

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
        pkt.noise_floor = rx_ctrl->noise_floor;
#endif

        // Copy CSI data
        pkt.csi_len = info->len > 640 ? 640 : info->len;
        memcpy(pkt.csi_data, info->buf, pkt.csi_len);

        // Non-blocking queue send (drop if full)
        if (xQueueSend(s_csi_queue, &pkt, 0) != pdTRUE) {
            // Queue full, packet dropped
            static uint32_t drop_count = 0;
            if (++drop_count % 100 == 0) {
                ESP_LOGW(TAG, "Queue full, %lu packets dropped", drop_count);
            }
        }
    }

#ifdef CONFIG_WAVIRA_DEBUG_SERIAL
    // Also output to serial for debugging
#endif

#else
    // Serial Mode: Output to UART
    uint32_t rx_id = s_packet_count;
#ifdef CONFIG_CSI_TRIGGER_ESPNOW
    rx_id = *(uint32_t *)(info->payload + 15);
#endif

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
    if (!s_packet_count) {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data\n");
    }

    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d",
               rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate,
               rx_ctrl->noise_floor, phy_info->fft_gain, phy_info->agc_gain, rx_ctrl->channel,
               rx_ctrl->timestamp, rx_ctrl->sig_len, rx_ctrl->rx_state);
#else
    if (!s_packet_count) {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data\n");
    }

    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
               rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
               rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
               rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
               rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt, rx_ctrl->channel, rx_ctrl->secondary_channel,
               rx_ctrl->timestamp, rx_ctrl->ant, rx_ctrl->sig_len, rx_ctrl->rx_state);
#endif

#if CONFIG_IDF_TARGET_ESP32C5 && CSI_FORCE_LLTF
    ets_printf(",%d,%d,\"[%d", (info->len - 2) / 2, info->first_word_invalid,
               (int16_t)(((int16_t)info->buf[1]) << 12) >> 4 | (uint8_t)info->buf[0]);
    for (int i = 2; i < (info->len - 2); i += 2) {
        ets_printf(",%d", (int16_t)(((int16_t)info->buf[i + 1]) << 12) >> 4 | (uint8_t)info->buf[i]);
    }
#else
    ets_printf(",%d,%d,\"[%d", info->len, info->first_word_invalid, info->buf[0]);
    for (int i = 1; i < info->len; i++) {
        ets_printf(",%d", info->buf[i]);
    }
#endif
    ets_printf("]\"\n");

#endif // CONFIG_CSI_OUTPUT_HTTP || CONFIG_CSI_OUTPUT_MQTT

    s_packet_count++;
}

// =============================================================================
// Wi-Fi Initialization (Serial Mode - ESP-NOW)
// =============================================================================

#if !defined(CONFIG_CSI_OUTPUT_HTTP) && !defined(CONFIG_CSI_OUTPUT_MQTT)
static void wifi_init_espnow(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

#if CONFIG_IDF_TARGET_ESP32C5
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
        .ghz_5g = CONFIG_WIFI_5G_PROTOCOL
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
        .ghz_5g = CONFIG_WIFI_5G_BANDWIDTHS
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#elif CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = { .ghz_2g = CONFIG_WIFI_2G_PROTOCOL };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = { .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#else
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());
#endif

#if CONFIG_IDF_TARGET_ESP32 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32S3
    ESP_ERROR_CHECK(esp_wifi_config_espnow_rate(ESP_IF_WIFI_STA, CONFIG_ESP_NOW_RATE));
#endif
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
    if ((CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20)
        || (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_5G_ONLY && CONFIG_WIFI_5G_BANDWIDTHS == WIFI_BW_HT20)) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#else
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#endif

#ifdef CONFIG_CSI_TRIGGER_ESPNOW
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
#endif
}

#if CONFIG_IDF_TARGET_ESP32C5
static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = CONFIG_ESP_NOW_RATE,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));
}
#endif
#endif // !CONFIG_CSI_OUTPUT_HTTP && !CONFIG_CSI_OUTPUT_MQTT

// =============================================================================
// CSI Initialization
// =============================================================================

static void wifi_csi_init(void *ctx)
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

#if CONFIG_IDF_TARGET_ESP32C5
    wifi_csi_config_t csi_config = {
        .enable = true,
        .acquire_csi_legacy = false,
        .acquire_csi_force_lltf = CSI_FORCE_LLTF,
        .acquire_csi_ht20 = true,
        .acquire_csi_ht40 = true,
        .acquire_csi_vht = false,
        .acquire_csi_su = false,
        .acquire_csi_mu = false,
        .acquire_csi_dcm = false,
        .acquire_csi_beamformed = false,
        .acquire_csi_he_stbc_mode = 2,
        .val_scale_cfg = 0,
        .dump_ack_en = false,
        .reserved = false
    };
#elif CONFIG_IDF_TARGET_ESP32C6
    wifi_csi_config_t csi_config = {
        .enable = true,
        .acquire_csi_legacy = false,
        .acquire_csi_ht20 = true,
        .acquire_csi_ht40 = true,
        .acquire_csi_su = true,
        .acquire_csi_mu = true,
        .acquire_csi_dcm = true,
        .acquire_csi_beamformed = true,
        .acquire_csi_he_stbc = 2,
        .val_scale_cfg = false,
        .dump_ack_en = false,
        .reserved = false
    };
#else
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = false,
    };
#endif

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, ctx));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI initialized");
}

// =============================================================================
// Main Entry Point
// =============================================================================

void app_main(void)
{
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "  Wavira CSI Receiver v1.0.0");
    ESP_LOGI(TAG, "==============================================");
#ifdef CONFIG_CSI_OUTPUT_HTTP
    ESP_LOGI(TAG, "  Mode: HTTP (Wi-Fi)");
    ESP_LOGI(TAG, "  Device ID: %s", CONFIG_WAVIRA_DEVICE_ID);
    ESP_LOGI(TAG, "  Server: %s", CONFIG_WAVIRA_SERVER_URL);
#elif defined(CONFIG_CSI_OUTPUT_MQTT)
    ESP_LOGI(TAG, "  Mode: MQTT (Wi-Fi)");
    ESP_LOGI(TAG, "  Device ID: %s", CONFIG_WAVIRA_DEVICE_ID);
    ESP_LOGI(TAG, "  Broker: %s:%d", CONFIG_WAVIRA_MQTT_BROKER_URL, CONFIG_WAVIRA_MQTT_PORT);
#else
    ESP_LOGI(TAG, "  Mode: Serial (UART)");
#endif
    ESP_LOGI(TAG, "==============================================");

    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Start LED task
    xTaskCreate(&led_task, "led_task", 4096, NULL, 5, NULL);

#ifdef CONFIG_CSI_OUTPUT_HTTP
    // HTTP Mode
    s_boot_time_ms = esp_timer_get_time() / 1000;

    // Create CSI packet queue for buffering
    s_csi_queue = xQueueCreate(CONFIG_WAVIRA_BUFFER_SIZE, sizeof(csi_packet_t));
    if (!s_csi_queue) {
        ESP_LOGE(TAG, "Failed to create CSI queue");
        return;
    }

    // Connect to Wi-Fi
    wifi_init_sta();

    // Get AP BSSID for CSI filtering (router mode)
#ifdef CONFIG_CSI_TRIGGER_ROUTER
    wifi_ap_record_t ap_info;
    ESP_ERROR_CHECK(esp_wifi_sta_get_ap_info(&ap_info));
    wifi_csi_init(ap_info.bssid);
    wifi_ping_router_start();
#else
    wifi_csi_init(NULL);
#endif

    // Start HTTP sender task
    xTaskCreate(&http_sender_task, "http_sender", 16384, NULL, 5, NULL);

#elif defined(CONFIG_CSI_OUTPUT_MQTT)
    // MQTT Mode
    s_boot_time_ms = esp_timer_get_time() / 1000;

    // Create CSI packet queue for buffering
    s_csi_queue = xQueueCreate(CONFIG_WAVIRA_BUFFER_SIZE, sizeof(csi_packet_t));
    if (!s_csi_queue) {
        ESP_LOGE(TAG, "Failed to create CSI queue");
        return;
    }

    // Connect to Wi-Fi (reuse HTTP mode Wi-Fi init)
    wifi_init_sta();

    // Initialize MQTT client
    mqtt_init();

    // Get AP BSSID for CSI filtering (router mode)
#ifdef CONFIG_CSI_TRIGGER_ROUTER
    wifi_ap_record_t ap_info;
    ESP_ERROR_CHECK(esp_wifi_sta_get_ap_info(&ap_info));
    wifi_csi_init(ap_info.bssid);
    wifi_ping_router_start();
#else
    wifi_csi_init(NULL);
#endif

    // Start MQTT sender task
    xTaskCreate(&mqtt_sender_task, "mqtt_sender", 16384, NULL, 5, NULL);

#else
    // Serial Mode (ESP-NOW)
    wifi_init_espnow();

#if CONFIG_IDF_TARGET_ESP32C5
    esp_now_peer_info_t peer = {
        .channel = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx = WIFI_IF_STA,
        .encrypt = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };
    wifi_esp_now_init(peer);
#endif

    wifi_csi_init(NULL);
#endif

    ESP_LOGI(TAG, "Initialization complete, waiting for CSI data...");
}
