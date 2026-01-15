/*
 * SPDX-FileCopyrightText: 2025 Wavira Project
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef WAVIRA_OTA_H
#define WAVIRA_OTA_H

#include "esp_err.h"
#include "mqtt_client.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief OTA update status codes
 */
typedef enum {
    OTA_STATUS_IDLE = 0,           /**< No OTA in progress */
    OTA_STATUS_DOWNLOADING,         /**< Downloading firmware */
    OTA_STATUS_VERIFYING,           /**< Verifying firmware */
    OTA_STATUS_APPLYING,            /**< Applying firmware */
    OTA_STATUS_SUCCESS,             /**< OTA completed successfully */
    OTA_STATUS_FAILED,              /**< OTA failed */
    OTA_STATUS_ROLLBACK,            /**< Rolling back to previous firmware */
} ota_status_t;

/**
 * @brief OTA progress information
 */
typedef struct {
    ota_status_t status;            /**< Current OTA status */
    uint32_t total_size;            /**< Total firmware size in bytes */
    uint32_t downloaded_size;       /**< Downloaded size in bytes */
    uint8_t progress_percent;       /**< Download progress (0-100) */
    char error_msg[64];             /**< Error message if failed */
} ota_progress_t;

/**
 * @brief Initialize OTA subsystem
 *
 * Sets up the OTA module and registers MQTT subscription handlers.
 * Must be called after MQTT client is initialized.
 *
 * @param mqtt_client Handle to initialized MQTT client
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t ota_init(esp_mqtt_client_handle_t mqtt_client);

/**
 * @brief Start OTA update from URL
 *
 * Begins downloading and applying firmware from the specified URL.
 * Progress is reported via MQTT topic: wavira/device/{id}/ota/progress
 *
 * @param url HTTPS URL of the firmware binary
 * @return ESP_OK if OTA started, error code otherwise
 */
esp_err_t ota_start(const char *url);

/**
 * @brief Get current OTA progress
 *
 * @param progress Pointer to store progress information
 * @return ESP_OK on success, ESP_ERR_INVALID_STATE if no OTA in progress
 */
esp_err_t ota_get_progress(ota_progress_t *progress);

/**
 * @brief Mark current firmware as valid
 *
 * Should be called after successful boot to confirm firmware is working.
 * Prevents automatic rollback on next boot.
 *
 * @return ESP_OK on success
 */
esp_err_t ota_mark_valid(void);

/**
 * @brief Trigger rollback to previous firmware
 *
 * Switches to the previous OTA partition and reboots.
 *
 * @return Does not return on success, error code on failure
 */
esp_err_t ota_rollback(void);

/**
 * @brief Check if rollback is available
 *
 * @return true if previous firmware is available for rollback
 */
bool ota_rollback_available(void);

/**
 * @brief Get current firmware version
 *
 * @return Firmware version string from app descriptor
 */
const char *ota_get_version(void);

/**
 * @brief Handle MQTT OTA command message
 *
 * Called internally when OTA command is received via MQTT.
 *
 * @param topic MQTT topic
 * @param data Message payload
 * @param data_len Payload length
 */
void ota_handle_mqtt_command(const char *topic, const char *data, int data_len);

#ifdef __cplusplus
}
#endif

#endif /* WAVIRA_OTA_H */
