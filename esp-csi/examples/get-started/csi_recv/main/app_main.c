/*
 * SPDX-FileCopyrightText: 2025 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
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
#include "mqtt_client.h"
#include "cJSON.h"

static const char *TAG = "wavira_csi";

typedef struct {
    uint32_t seq;
    uint8_t mac[6];
    int8_t rssi;
    int64_t timestamp_ms;
    uint16_t csi_len;
    int8_t csi_data[384];
} csi_packet_t;

static EventGroupHandle_t s_wifi_event_group;
static QueueHandle_t s_csi_queue;
static esp_mqtt_client_handle_t s_mqtt_client = NULL;
static volatile bool s_wifi_connected = false;
static volatile bool s_mqtt_connected = false;
static uint32_t s_packet_count = 0;

#define WIFI_CONNECTED_BIT BIT0

static void wifi_csi_init(void *ctx)
{
    esp_wifi_set_promiscuous(true);
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || !s_mqtt_connected || !s_csi_queue) return;

    static int64_t last_time = 0;
    int64_t now = esp_timer_get_time() / 1000;
    if (now - last_time < 200) return; // Limit to 5Hz for stability
    last_time = now;

    static csi_packet_t pkt;
    pkt.seq = s_packet_count++;
    memcpy(pkt.mac, info->mac, 6);
    pkt.rssi = info->rx_ctrl.rssi;
    pkt.timestamp_ms = now;
    pkt.csi_len = info->len > 384 ? 384 : info->len;
    memcpy(pkt.csi_data, info->buf, pkt.csi_len);

    xQueueSend(s_csi_queue, &pkt, 0);
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_mqtt_event_handle_t event = event_data;
    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            ESP_LOGI(TAG, "MQTT connected");
            s_mqtt_connected = true;
            char topic[128];
            snprintf(topic, sizeof(topic), "wavira/device/%s/status", CONFIG_WAVIRA_DEVICE_ID);
            esp_mqtt_client_publish(s_mqtt_client, topic, "{\"status\":\"online\"}", 0, 1, 1);
            break;
        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT disconnected");
            s_mqtt_connected = false;
            break;
        default:
            break;
    }
}

static void mqtt_sender_task(void *pvParameter)
{
    csi_packet_t pkt;
    char topic[128];
    snprintf(topic, sizeof(topic), "wavira/csi/%s", CONFIG_WAVIRA_DEVICE_ID);

    ESP_LOGI(TAG, "MQTT sender task started on topic: %s", topic);

    while (1) {
        if (xQueueReceive(s_csi_queue, &pkt, portMAX_DELAY)) {
            if (!s_mqtt_connected) continue;

            cJSON *root = cJSON_CreateObject();
            cJSON_AddStringToObject(root, "id", CONFIG_WAVIRA_DEVICE_ID);
            cJSON_AddNumberToObject(root, "ts", pkt.timestamp_ms);
            cJSON_AddNumberToObject(root, "rssi", pkt.rssi);
            
            cJSON *data = cJSON_CreateArray();
            for (int i = 0; i < pkt.csi_len; i++) {
                cJSON_AddItemToArray(data, cJSON_CreateNumber(pkt.csi_data[i]));
            }
            cJSON_AddItemToObject(root, "data", data);

            char *json_str = cJSON_PrintUnformatted(root);
            if (json_str) {
                esp_mqtt_client_publish(s_mqtt_client, topic, json_str, 0, 0, 0);
                free(json_str);
            }
            cJSON_Delete(root);
        }
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_wifi_connected = false;
        esp_wifi_connect();
    } else if (event_id == IP_EVENT_STA_GOT_IP) {
        s_wifi_connected = true;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

void app_main(void)
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Initialize Network
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_wifi_event_group = xEventGroupCreate();
    esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));
    
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = CONFIG_WAVIRA_WIFI_SSID,
            .password = CONFIG_WAVIRA_WIFI_PASSWORD,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Wait for Wi-Fi
    ESP_LOGI(TAG, "Waiting for Wi-Fi...");
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdFALSE, portMAX_DELAY);
    ESP_LOGI(TAG, "Wi-Fi connected");

    // Initialize CSI
    s_csi_queue = xQueueCreate(10, sizeof(csi_packet_t));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    wifi_csi_init(NULL);

    // Initialize MQTT
    esp_mqtt_client_config_t mqtt_cfg = {};
    mqtt_cfg.broker.address.uri = CONFIG_WAVIRA_MQTT_BROKER_URL;
    
    s_mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(s_mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(s_mqtt_client);

    // Start background task
    xTaskCreate(&mqtt_sender_task, "mqtt_sender", 16384, NULL, 5, NULL);

    ESP_LOGI(TAG, "System initialized");
}