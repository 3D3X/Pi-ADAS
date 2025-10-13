from rpi5_ws2812.ws2812 import Color, WS2812SpiDriver
import time

if __name__ == "__main__":
    # Inicjalizacja paska LED (100 diod, SPI channel 0)
    strip = WS2812SpiDriver(spi_bus=0, spi_device=0, led_count=8).get_strip()
    while True:
        strip.set_all_pixels(Color(255, 0, 0))  # Czerwony
        strip.show()
        time.sleep(1)
        strip.set_all_pixels(Color(0, 255, 0))  # Zielony
        strip.show()
        time.sleep(1)
