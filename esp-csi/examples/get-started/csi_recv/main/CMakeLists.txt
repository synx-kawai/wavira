set(COMPONENT_REQUIRES
    nvs_flash
    esp_wifi
    esp_netif
    esp_event
    esp_now
    driver
)

# Add HTTP mode dependencies if enabled
if(CONFIG_CSI_OUTPUT_HTTP)
    list(APPEND COMPONENT_REQUIRES
        esp_http_client
        json
        lwip
        esp_ping
    )
endif()

idf_component_register(SRC_DIRS "."
                       INCLUDE_DIRS "."
                       REQUIRES ${COMPONENT_REQUIRES})
