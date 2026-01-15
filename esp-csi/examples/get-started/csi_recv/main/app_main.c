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
#include "lwip/sockets.h"

#if CONFIG_IDF_TARGET_ESP32S3
#include "led_strip.h"
#endif

static const char *TAG = "wavira_csi";

// LED Configuration
// ESP32: GPIO2 (built-in blue LED) - simple GPIO
// ESP32-S3 DevKitC: GPIO48 (addressable RGB LED) - needs led_strip driver
// XIAO ESP32S3: GPIO21 (built-in yellow LED) - simple GPIO
#if CONFIG_IDF_TARGET_ESP32S3
// Use GPIO21 for XIAO ESP32S3 (simple GPIO LED, not RGB)
#define LED_GPIO 21
#define LED_USE_SIMPLE_GPIO 1
// Uncomment below for ESP32-S3-DevKitC with RGB LED on GPIO48
// #define LED_GPIO 48
// #define LED_USE_RGB_STRIP 1
#else
#define LED_GPIO 2
#define LED_USE_SIMPLE_GPIO 1
#endif

#ifdef LED_USE_RGB_STRIP
static led_strip_handle_t s_led_strip = NULL;
#endif

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
static TaskHandle_t s_led_task_handle = NULL;
static volatile bool s_wifi_connected = false;
static volatile bool s_mqtt_connected = false;
static uint32_t s_packet_count = 0;
static uint32_t s_gateway_ip = 0;

#define WIFI_CONNECTED_BIT BIT0

// LED State
typedef enum {
    LED_STATE_OFF = 0,
    LED_STATE_BOOTING,      // Red - startup
    LED_STATE_WIFI_WAIT,    // Yellow blink - waiting for WiFi
    LED_STATE_MQTT_WAIT,    // Blue blink - waiting for MQTT
    LED_STATE_CONNECTED,    // Green - normal operation
    LED_STATE_SENDING,      // Green pulse - data transmission
    LED_STATE_ERROR,        // Red blink - error
} led_state_t;

static volatile led_state_t s_led_state = LED_STATE_OFF;

// Forward declarations for LED pulse functions
static void led_pulse_tx(void);
static void led_pulse_rx(void);

// LED helper functions
#ifdef LED_USE_RGB_STRIP
static void led_set_rgb(uint8_t r, uint8_t g, uint8_t b)
{
    if (s_led_strip) {
        led_strip_set_pixel(s_led_strip, 0, r, g, b);
        led_strip_refresh(s_led_strip);
    }
}
#else
// Simple GPIO LED - use brightness threshold
static void led_set_rgb(uint8_t r, uint8_t g, uint8_t b)
{
    gpio_set_level(LED_GPIO, (r + g + b) > 0 ? 1 : 0);
}
#endif

static void led_set(int on)
{
    gpio_set_level(LED_GPIO, on ? 1 : 0);
}

static void led_init(void)
{
#ifdef LED_USE_RGB_STRIP
    // RGB LED strip initialization (ESP32-S3-DevKitC)
    led_strip_config_t strip_config = {
        .strip_gpio_num = LED_GPIO,
        .max_leds = 1,
    };
    led_strip_rmt_config_t rmt_config = {
        .resolution_hz = 10 * 1000 * 1000,  // 10MHz
        .flags.with_dma = false,
    };
    esp_err_t err = led_strip_new_rmt_device(&strip_config, &rmt_config, &s_led_strip);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "LED strip init failed: %s", esp_err_to_name(err));
        return;
    }
    led_strip_clear(s_led_strip);
    ESP_LOGI(TAG, "RGB LED strip initialized on GPIO%d", LED_GPIO);

    // LED test sequence: Red -> Green -> Blue (300ms each)
    ESP_LOGI(TAG, "LED test: Red");
    led_set_rgb(64, 0, 0);
    vTaskDelay(pdMS_TO_TICKS(300));
    ESP_LOGI(TAG, "LED test: Green");
    led_set_rgb(0, 64, 0);
    vTaskDelay(pdMS_TO_TICKS(300));
    ESP_LOGI(TAG, "LED test: Blue");
    led_set_rgb(0, 0, 64);
    vTaskDelay(pdMS_TO_TICKS(300));
    led_set_rgb(0, 0, 0);
    ESP_LOGI(TAG, "LED test complete");
#else
    // Simple GPIO LED initialization (XIAO ESP32S3, ESP32)
    gpio_reset_pin(LED_GPIO);
    gpio_set_direction(LED_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level(LED_GPIO, 0);
    ESP_LOGI(TAG, "LED initialized on GPIO%d", LED_GPIO);

    // LED test: 3 quick blinks
    ESP_LOGI(TAG, "LED test: 3 blinks");
    for (int i = 0; i < 3; i++) {
        gpio_set_level(LED_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(150));
        gpio_set_level(LED_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(150));
    }
    ESP_LOGI(TAG, "LED test complete");
#endif
}

// CSI trigger task - sends UDP packets to gateway to trigger CSI measurements
static void csi_trigger_task(void *pvParameter)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Failed to create UDP socket");
        vTaskDelete(NULL);
        return;
    }

    struct sockaddr_in dest_addr;
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(12345);  // Arbitrary port
    dest_addr.sin_addr.s_addr = s_gateway_ip;

    const char *trigger_data = "CSI";
    ESP_LOGI(TAG, "CSI trigger task started (10Hz UDP to gateway)");

    int led_counter = 0;
    while (s_wifi_connected || !s_mqtt_connected) {
        // Continue running while system is active
        if (s_wifi_connected) {
            sendto(sock, trigger_data, 3, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
            // Flash LED every 5th packet (~2Hz visible blink for TX activity)
            if (++led_counter >= 5) {
                led_pulse_tx();
                led_counter = 0;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(100));  // 10Hz
    }

    // Clean up socket before task exit (defensive programming)
    close(sock);
    ESP_LOGI(TAG, "CSI trigger task stopped, socket closed");
    vTaskDelete(NULL);
}

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
    if (now - last_time < 500) return; // Limit to 2Hz for high-latency stability
    last_time = now;

    // Flash LED on CSI receive (RX activity)
    led_pulse_rx();

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
            ESP_LOGW(TAG, "MQTT disconnected - attempting reconnect...");
            s_mqtt_connected = false;
            // Brief delay before reconnect to avoid rapid reconnect loops
            vTaskDelay(pdMS_TO_TICKS(2000));
            esp_mqtt_client_reconnect(s_mqtt_client);
            break;
        case MQTT_EVENT_ERROR:
            ESP_LOGE(TAG, "MQTT error event - type: %d", event->error_handle->error_type);
            if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                ESP_LOGE(TAG, "TCP transport error: 0x%x", event->error_handle->esp_tls_last_esp_err);
            } else if (event->error_handle->error_type == MQTT_ERROR_TYPE_CONNECTION_REFUSED) {
                ESP_LOGE(TAG, "Connection refused error: 0x%x", event->error_handle->connect_return_code);
            }
            break;
        default:
            break;
    }
}

// Timer-based LED status indicator (avoids task/WiFi conflicts)
// Pattern: LED on = connected, LED blinking = connecting/error
// Activity pattern: Brief flash on TX/RX like network switch LEDs
static esp_timer_handle_t s_led_timer = NULL;
static int s_led_blink_state = 0;
static int s_led_blink_interval = 500;  // ms
static volatile bool s_led_tx_pulse = false;  // TX activity (sending data)
static volatile bool s_led_rx_pulse = false;  // RX activity (receiving CSI)
static volatile int s_led_pulse_duration = 0; // Pulse duration counter

// LED pulse duration in timer ticks (each tick = timer interval)
#define LED_PULSE_TICKS 2  // ~40ms at 20ms timer = visible flash

static void led_timer_callback(void *arg)
{
    s_led_blink_state = !s_led_blink_state;

    if (!s_wifi_connected) {
        // WiFi not connected: slow blink (500ms)
        s_led_state = LED_STATE_WIFI_WAIT;
        gpio_set_level(LED_GPIO, s_led_blink_state);
        if (s_led_blink_interval != 500) {
            s_led_blink_interval = 500;
            esp_timer_stop(s_led_timer);
            esp_timer_start_periodic(s_led_timer, 500 * 1000);
        }
    } else if (!s_mqtt_connected) {
        // WiFi connected but MQTT not: fast blink (200ms)
        s_led_state = LED_STATE_MQTT_WAIT;
        gpio_set_level(LED_GPIO, s_led_blink_state);
        if (s_led_blink_interval != 200) {
            s_led_blink_interval = 200;
            esp_timer_stop(s_led_timer);
            esp_timer_start_periodic(s_led_timer, 200 * 1000);
        }
    } else {
        // MQTT connected: Network switch style LED activity
        // LED flashes briefly on TX or RX activity
        s_led_state = LED_STATE_CONNECTED;

        // Check for new activity pulses
        // Note: Read-modify-write on volatile flags. This is safe because:
        // 1. ESP32 timer callbacks run in ISR context on a single core
        // 2. The flags are only set (not read) from other contexts
        // 3. For multi-core scenarios, consider using atomic operations
        if (s_led_tx_pulse || s_led_rx_pulse) {
            s_led_pulse_duration = LED_PULSE_TICKS;
            s_led_tx_pulse = false;
            s_led_rx_pulse = false;
        }

        // LED ON during pulse duration, OFF otherwise
        if (s_led_pulse_duration > 0) {
            gpio_set_level(LED_GPIO, 1);
            s_led_pulse_duration--;
        } else {
            gpio_set_level(LED_GPIO, 0);
        }

        // Fast timer (20ms) for responsive activity indication
        if (s_led_blink_interval != 20) {
            s_led_blink_interval = 20;
            esp_timer_stop(s_led_timer);
            esp_timer_start_periodic(s_led_timer, 20 * 1000);
        }
    }
}

static void led_timer_init(void)
{
    esp_timer_create_args_t timer_args = {
        .callback = led_timer_callback,
        .name = "led_timer"
    };
    ESP_ERROR_CHECK(esp_timer_create(&timer_args, &s_led_timer));
    esp_timer_start_periodic(s_led_timer, 500 * 1000);  // Start with 500ms
    ESP_LOGI(TAG, "LED timer started");
}

// Call this to trigger LED pulse on TX (data transmission)
static void led_pulse_tx(void)
{
    s_led_tx_pulse = true;
}

// Call this to trigger LED pulse on RX (CSI data received)
static void led_pulse_rx(void)
{
    s_led_rx_pulse = true;
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
                // Flash LED on MQTT send (TX activity)
                led_pulse_tx();
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
    // Initialize LED
    led_init();
    led_set(0);

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

    // Disable WiFi power save mode to prevent disconnections
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    // External antenna configuration for XIAO ESP32S3 Plus
    // Note: XIAO ESP32S3 uses a simpler antenna selection without GPIO switch
    wifi_ant_config_t ant_config = {
        .rx_ant_mode = WIFI_ANT_MODE_ANT1,  // Use external antenna
        .tx_ant_mode = WIFI_ANT_MODE_ANT1,
        .rx_ant_default = WIFI_ANT_ANT1,
        .enabled_ant0 = 0,  // Disable internal PCB antenna
        .enabled_ant1 = 1,  // Enable external U.FL antenna
    };
    esp_err_t ant_err = esp_wifi_set_ant(&ant_config);
    if (ant_err == ESP_OK) {
        ESP_LOGI(TAG, "External antenna enabled");
    } else {
        ESP_LOGW(TAG, "Antenna config failed: %s (using default)", esp_err_to_name(ant_err));
    }

    // Start LED timer (timer-based approach avoids task/WiFi conflicts)
    led_timer_init();

    // Wait for Wi-Fi (LED task shows yellow blink)
    ESP_LOGI(TAG, "Waiting for Wi-Fi connection to %s...", CONFIG_WAVIRA_WIFI_SSID);
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_LOGI(TAG, "Wi-Fi connected!");

    // Initialize CSI
    s_csi_queue = xQueueCreate(10, sizeof(csi_packet_t));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    wifi_csi_init(NULL);

    // Initialize MQTT with high-latency tolerance settings
    esp_mqtt_client_config_t mqtt_cfg = {};
    mqtt_cfg.broker.address.uri = CONFIG_WAVIRA_MQTT_BROKER_URL;

    // Network settings optimized for NAT traversal and stability
    mqtt_cfg.network.timeout_ms = 30000;           // 30s network timeout
    mqtt_cfg.network.reconnect_timeout_ms = 5000;  // 5s before reconnect attempt
    mqtt_cfg.session.keepalive = 30;               // 30s keep-alive (shorter for NAT compatibility)
    mqtt_cfg.buffer.size = 2048;                   // Larger buffer for reliability
    mqtt_cfg.buffer.out_size = 2048;

    // Last Will and Testament for proper disconnect detection
    static char will_topic[128];
    static char will_msg[256];
    snprintf(will_topic, sizeof(will_topic), "wavira/device/%s/will", CONFIG_WAVIRA_DEVICE_ID);
    snprintf(will_msg, sizeof(will_msg), "{\"device_id\":\"%s\",\"status\":\"offline\",\"reason\":\"unexpected_disconnect\"}", CONFIG_WAVIRA_DEVICE_ID);
    mqtt_cfg.session.last_will.topic = will_topic;
    mqtt_cfg.session.last_will.msg = will_msg;
    mqtt_cfg.session.last_will.qos = 1;
    mqtt_cfg.session.last_will.retain = true;

    s_mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(s_mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(s_mqtt_client);

    // Start MQTT sender task
    xTaskCreate(&mqtt_sender_task, "mqtt_sender", 16384, NULL, 5, NULL);

    // Get gateway IP and start CSI trigger task
    esp_netif_ip_info_t ip_info;
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (netif && esp_netif_get_ip_info(netif, &ip_info) == ESP_OK) {
        s_gateway_ip = ip_info.gw.addr;
        ESP_LOGI(TAG, "Gateway IP: " IPSTR, IP2STR(&ip_info.gw));
        xTaskCreate(&csi_trigger_task, "csi_trigger", 4096, NULL, 4, NULL);
    } else {
        ESP_LOGE(TAG, "Failed to get gateway IP");
    }

    ESP_LOGI(TAG, "System initialized");
}
