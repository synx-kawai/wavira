set(COMPONENT_REQUIRES
    nvs_flash
    esp_wifi
    esp_netif
    esp_event
    espressif__esp-now
    driver
    mqtt
    lwip
)

idf_component_register(SRC_DIRS "."
                       INCLUDE_DIRS "."
                       REQUIRES ${COMPONENT_REQUIRES}
                       PRIV_REQUIRES json esp-tls)