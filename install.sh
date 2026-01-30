#!/bin/bash
# ============================================================
# Skrypt instalacyjny PI-ADAS
# Python 3.13.5 | Raspberry Pi 5 | Debian 13 (trixie)
# ============================================================

set -e  # Zatrzymaj przy pierwszym błędzie

echo "=================================================="
echo "  Instalacja środowiska PI-ADAS"
echo "  Python 3.13.5 | Raspberry Pi 5"
echo "=================================================="
echo ""

# Kolory dla outputu
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Sprawdź wersję Python
echo -e "${YELLOW}[1/7] Sprawdzanie wersji Python...${NC}"
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Wykryto Python: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.13.5" ]]; then
    echo -e "${YELLOW}⚠️  Ostrzeżenie: Wykryto Python $PYTHON_VERSION zamiast 3.13.5${NC}"
    echo "Kontynuuję instalację..."
fi
echo ""

# Aktualizacja systemu
echo -e "${YELLOW}[2/7] Aktualizacja systemu...${NC}"
sudo apt-get update
echo -e "${GREEN}✓ System zaktualizowany${NC}"
echo ""

# Instalacja zależności systemowych
echo -e "${YELLOW}[3/7] Instalacja zależności systemowych...${NC}"
sudo apt-get install -y \
    python3-pygame \
    python3-dev \
    build-essential \
    libopenblas-dev \
    git
echo -e "${GREEN}✓ Zależności systemowe zainstalowane${NC}"
echo ""

# Konfiguracja uprawnień
echo -e "${YELLOW}[4/7] Konfiguracja uprawnień użytkownika...${NC}"
sudo usermod -a -G spi,gpio,video $USER
sudo chmod 666 /dev/spidev* 2>/dev/null || true
echo -e "${GREEN}✓ Uprawnienia skonfigurowane${NC}"
echo -e "${YELLOW}⚠️  Wyloguj się i zaloguj ponownie aby zastosować zmiany grup!${NC}"
echo ""

# Usunięcie starego venv jeśli istnieje
if [ -d ".venv" ]; then
    echo -e "${YELLOW}[5/7] Usuwanie starego środowiska wirtualnego...${NC}"
    rm -rf .venv
    echo -e "${GREEN}✓ Stare środowisko usunięte${NC}"
else
    echo -e "${YELLOW}[5/7] Brak starego środowiska do usunięcia${NC}"
fi
echo ""

# Tworzenie nowego venv z dostępem do pakietów systemowych
echo -e "${YELLOW}[6/7] Tworzenie nowego środowiska wirtualnego...${NC}"
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
echo -e "${GREEN}✓ Środowisko wirtualne utworzone${NC}"
echo ""

# Aktualizacja pip
echo -e "${YELLOW}Aktualizacja pip...${NC}"
pip install --upgrade pip
echo ""

# Instalacja pakietów Python
echo -e "${YELLOW}[7/7] Instalacja pakietów Python...${NC}"
pip install \
    lgpio==0.2.2.0 \
    opencv-python==4.11.0.86 \
    numpy==1.26.4 \
    matplotlib>=3.8 \
    pandas-stubs \
    obd==0.7.3 \
    picamera2==0.3.25 \
    rpi5-ws2812==0.1.2

echo ""
echo -e "${GREEN}✓ Pakiety Python zainstalowane${NC}"
echo ""

# Instalacja TensorFlow Lite (opcjonalne)
echo -e "${YELLOW}Czy chcesz zainstalować TensorFlow Lite? (t/n)${NC}"
read -r response
if [[ "$response" =~ ^([tT][aA][kK]|[tT])$ ]]; then
    echo -e "${YELLOW}Instalacja tflite-runtime...${NC}"
    pip install tflite-runtime 2>/dev/null || {
        echo -e "${YELLOW}⚠️  tflite-runtime niedostępne dla Python 3.13${NC}"
        echo -e "${YELLOW}Instaluję pełny TensorFlow (może zająć kilka minut)...${NC}"
        pip install tensorflow
    }
    echo -e "${GREEN}✓ TensorFlow zainstalowany${NC}"
else
    echo "Pominięto instalację TensorFlow"
fi
echo ""

# Weryfikacja instalacji
echo -e "${YELLOW}Weryfikacja instalacji...${NC}"
python3 << EOF
try:
    import cv2
    import numpy as np
    import lgpio
    import pygame
    import obd
    from picamera2 import Picamera2
    from rpi5_ws2812.ws2812 import Color, WS2812SpiDriver
    
    print("✅ Wszystkie moduły załadowane pomyślnie!")
    print(f"   OpenCV: {cv2.__version__}")
    print(f"   NumPy: {np.__version__}")
    print(f"   Pygame: {pygame.__version__}")
except ImportError as e:
    print(f"❌ Błąd importu: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================================="
    echo "  ✓ Instalacja zakończona pomyślnie!"
    echo "==================================================${NC}"
    echo ""
    echo "Następne kroki:"
    echo "1. Wyloguj się i zaloguj ponownie (aby zastosować uprawnienia grup)"
    echo "2. Aktywuj środowisko: source .venv/bin/activate"
    echo "3. Uruchom program: python3 PI-ADAS.py"
    echo ""
else
    echo ""
    echo -e "${RED}=================================================="
    echo "  ❌ Instalacja zakończona z błędami"
    echo "==================================================${NC}"
    exit 1
fi
