######## Advanced Driver-Assistance System for Raspberry Pi #########
#
# Author: Kostantinos Katsanos
# Date: 9/22 - 2/23
# Description:
# This program takes a feed from the camera or videofile and does lane detection using houghlines and
# object collission, car departure and run red light detection using bonding boxes from mobile_net_v3_small
# with TensorFlow Lite
# Object Detection and classification code is based on
# "Webcam Object Detection Using Tensorflow-trained Classifier" from Evan Juras & Ethan Dell

import cv2
import numpy as np
import os
import multiprocessing
os.environ["QT_QPA_PLATFORM"] = "xcb"
import argparse
import time
from threading import Thread
import importlib.util
import datetime
from pygame import mixer
import obd
import pickle
import math
import time
from picamera2 import Picamera2
from rpi5_ws2812.ws2812 import Color, WS2812SpiDriver
import json
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

# Global Config Variables
RaspberryPi = True
debugDisplay = True
videoFileMode = False
saveFramestoDisk = False
outdir = "images"
camOffset = 0
fi = []
fc = []
ft = []
show_perf_popup = False
button_rect = (10, 10, 150, 50)
# Pi Variables
buzzerPin = 13  # Configure the GPIO pin for the buzzer speaker
turnPin = (
    25  # Configure the GPIO pin for the turn signal input (hall effect/level converter)
)
buttonPin = 16


class AsyncLoadingAnimation(Thread):
    def __init__(self, num_leds=8, brightness=0.1):
        super().__init__()
        try:
            self.strip = WS2812SpiDriver(
                spi_bus=0, spi_device=0, led_count=num_leds
            ).get_strip()
            self.num_leds = num_leds
            self.brightness = brightness
            self.running = True
            self.phase = "start"
            self.daemon = True
            self.clear()
        except Exception as e:
            print(f"Błąd inicjalizacji LED: {e}")
            self.running = False

    def clear(self):
        try:
            if hasattr(self, "strip"):
                self.strip.set_all_pixels(Color(0, 0, 0))
                self.strip.show()
        except Exception as e:
            print(f"Błąd czyszczenia LED: {e}")

    def _set_pixel_color(self, pixel_num, r, g, b):
        try:
            if hasattr(self, "strip") and 0 <= pixel_num < self.num_leds:
                r = int(r * self.brightness)
                g = int(g * self.brightness)
                b = int(b * self.brightness)
                self.strip._pixels[pixel_num] = Color(r, g, b)
        except Exception as e:
            print(f"Błąd ustawiania koloru piksela: {e}")

    def _update_strip(self):
        try:
            if hasattr(self, "strip"):
                self.strip.show()
        except Exception as e:
            print(f"Błąd aktualizacji LED: {e}")

    def run(self):
        while self.running:
            try:
                if self.phase == "start":
                    color = (0, 0, 255)
                elif self.phase == "obd":
                    color = (255, 255, 0)
                else:  # camera
                    color = (0, 255, 0)

                r, g, b = color
                for i in range(self.num_leds):
                    if not self.running:
                        break
                    self.clear()
                    self._set_pixel_color(i, r, g, b)
                    self._update_strip()
                    time.sleep(0.1)

                for i in range(self.num_leds - 2, -1, -1):
                    if not self.running:
                        break
                    self.clear()
                    self._set_pixel_color(i, r, g, b)
                    self._update_strip()
                    time.sleep(0.1)
            except Exception as e:
                print(f"Błąd w animacji LED: {e}")
                time.sleep(0.5)

    def set_phase(self, phase):
        self.phase = phase

    def stop(self):
        self.running = False
        self.clear()


if RaspberryPi:
    import lgpio

    # from rpi_ws281x import Adafruit_NeoPixel, Color
    print("Inicjalizacja systemu, łączenie z OBD...")
    loading_animation = AsyncLoadingAnimation(num_leds=8)
    animation_thread = loading_animation.start()


# Video File Variables
cropoffset = 0
nameOfFile = ""
skip = 0
frameskip = 3
frameByframe = False


# ODBII Variables
accelN = 5
carAccel = 0
carSpeed = 0
tsign_left = False
tsign_right = False

# Lane Variables
starting_horizon_Ratio = (
    3 / 5
)  # (%)Starting horizon line persentage of image default is center
glassGradStreght = 65  # (0-255) Strenght of gradient that counteracts the brightness change of a slopping windshield
hood = 10  # (in pixels) Space at the bottom of image taken by the hood of the car
fov = (
    -250
)  # (in pixels) Starting point of the lane recognition based on the field of view of the camera
memoryframes = (
    10  # (in frames) Frames that a recognised line will be kept for feture calculations
)
carsize = 500

wheelOff = 50
newLineW = 0.8
smothBrightfact = 0.2
smothLaneCenter = 0.4
laneLockThresh = 20
lane_depth = 1.15
brightsens = 10


# Object Colision Variables
inPathSens = 1
weight_size = 0.9
weight_center = 1
newObjThresh = 40
CollisionSens = 0.05
trackingSens = 0.03
trackingoffThresh = 7
CollisionThresh = 9000
departureSens = 20
departureThresh = 30000


# Light Variables
# red
# c270
# (hMin = 0 , sMin = 40, vMin = 230), (hMax = 30 , sMax = 255, vMax = 255)

# C920
# INVERTED BGR (hMin = 100 , sMin = 60, vMin = 200), (hMax = 121 , sMax = 255, vMax = 255)
color_MINr = np.array([100, 60, 190], np.uint8)
color_MAXr = np.array([121, 255, 255], np.uint8)

# green (hMin = 74 , sMin = 51, vMin = 0), (hMax = 95 , sMax = 255, vMax = 255)
# (hMin = 83 , sMin = 11, vMin = 255), (hMax = 101 , sMax = 255, vMax = 255)
# c920
# INVERTED BGR (hMin = 0 , sMin = 40, vMin = 220), (hMax = 38 , sMax = 255, vMax = 255)
color_MINg = np.array([30, 60, 190], np.uint8)
color_MAXg = np.array([40, 255, 255], np.uint8)

lightthresh = 5
# lightlimit = 150
lightChangeThresh = 4  # actualy is 2x+1 frames of threshold
egdeThresh = 70
accelthresh = -1
minDistLight = 2
timeDistLight = 1


"""
# TODO Global
#  !!Add STOP signs
#  control window
#  track lights if model fails  
#
#  -------Done------
#  15/2------track objects if model fails  
#  15/1-----thread lanes
#  9/1------Create classes
# TODO LANE DEPARDURE
#  ???also take into account change of acceleration and direction
#  add threshhold for lines
#  Fuse with car detection:
#       make mask for coi
#       make center based on car if no lines are detected

#  -----Done------
#  25/1--failed---try canny thresholding and filter specific angles 
#  15/2---- add offset (camOff)
#  19/12------Make roi and car path point to center an remove the fluf with a mask
#  2/12-------make hsv threshold be based on road sarface brightness 
#  11/11------make fallback horizon auto average .  new_avg_horizon=horizon+abs(horizon-newhorizon)*weight (weight=0.x)
#  11/11------make a threshold for enabling warning
#  10/11---------- make lines persistent and more stable(by averaging many frames)
#  10/11-----------fix threshholds for detected lines at extreme angles or vertical
#  -----------add intersect with one line and horizon to find lane center
#  -----------fix blinking dashed lines
#  -----------add lane departure warning
#  -----------make image have less points and faster to proccess lines

# TODO CAR COLISION,DEPARTURE AND RUN LIGHT 
#  ! Compile my own lite mobilenet model using bdd100k
#  make inpath choose height or width witchever greater
#  maybe add the other double lights and choose them based on offcenter

#  ----------Done-----------
#  28/1-----Add Leading car departure --change to rate of change many frame average and not diference per frame -- add into account current speed
#  8/1----FIX Traffic light image staying in memory
#  8/1----Bug showing Red above green sometimes
#  2/1---------Add run Red light warning
#  21/12-------If car box too big or sides are on the edges then ignore collision
#  10/11-------Check if lights side by side and combine them into one double light  
#  10/11-------Maybe make non static values for trafic lights
#  fixed 10,11/11-------Average lights over many frames
"""



show_settings = False # Zmienna sterująca widocznością suwaków
button_rect = (10, 10, 150, 50) # x, y, szerokość, wysokość przycisku
CollisionSens = 0.05
CollisionThresh = 9000
departureSens = 20
minDistLight = 2
accelN = 5

def nothing(x):
    pass

def init_trackbars():
    cv2.namedWindow("settings", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("CollisionSens", "settings", 5, 20, nothing) # domyślnie 0.05
    cv2.createTrackbar("CollisionThresh", "settings", 9, 20, nothing) # domyślnie 9000
    cv2.createTrackbar("departureSens", "settings", 20, 50, nothing)
    cv2.createTrackbar("minDistLight", "settings", 2, 20, nothing)
    cv2.createTrackbar("accelN", "settings", 5, 15, nothing)

def on_touch(event, x, y, flags, param):
    global show_settings
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = button_rect
        if bx <= x <= bx + bw and by <= y <= by + bh:
            show_settings = not show_settings
            if show_settings:
                init_trackbars()
            else:
                try:
                    cv2.destroyWindow("settings")
                except:
                    pass

if RaspberryPi:

    class hallSens:

        def __init__(self):
            self.h = lgpio.gpiochip_open(4)  # Otwórz chip GPIO
            lgpio.gpio_claim_input(self.h, turnPin)  # Ustaw pin jako wejście

        def setTurn(self):
            global tsign_left, tsign_right
            if lgpio.gpio_read(self.h, turnPin):
                tsign_left = True
                tsign_right = True
            else:
                tsign_left = False
                tsign_right = False

        def cleanup(self):
            lgpio.gpiochip_close(self.h)  # Zwolnij zasoby GPIO

    class statLED:
        def __init__(self):
            self.strip = WS2812SpiDriver(
                spi_bus=0, spi_device=0, led_count=8
            ).get_strip()
            self.brightness = 0.05
            self.clear_leds()

            # Flagi stanu
            self.turnl = 0
            self.leftlane = 0
            self.carlock = 0
            self.rightlane = 0
            self.light = 0
            self.turnr = 0
            self.collision = 0

            # Predefiniowane kolory z uwzględnieniem jasności
            self.yellow = self._apply_brightness((255, 255, 0))
            self.green = self._apply_brightness((0, 255, 0))
            self.blue = self._apply_brightness((0, 0, 255))
            self.red = self._apply_brightness((255, 0, 0))
            self.black = (0, 0, 0)

            # Stan diod
            self.led_states = [self.black] * 8
            self.changed = False

            self.update_leds()

        def clear_leds(self):
            self.strip.set_all_pixels(Color(0, 0, 0))
            self.strip.show()

        def _apply_brightness(self, color):
            return tuple(int(c * self.brightness) for c in color)

        def set_state(self, **kwargs):
            """Ustawia flagi stanu bez natychmiastowej aktualizacji diod"""
            updated = False
            for key, value in kwargs.items():
                if hasattr(self, key) and getattr(self, key) != value:
                    setattr(self, key, value)
                    updated = True

            if updated:
                self.changed = True

        def update_leds(self):
            """Aktualizuje stan diod z uwzględnieniem priorytetów alarmów"""
            new_states = [self.black] * 8

            # --- POZIOM 1: ALARMY KRYTYCZNE (Nadpisują wszystko) ---
            if getattr(self, 'collision', 0) or getattr(self, 'run_red', 0):
                # Świecimy wszystko na jaskrawy czerwony (bez przyciemniania)
                alert_red = (255, 0, 0) 
                new_states = [alert_red] * 8

            # --- POZIOM 2: OSTRZEŻENIA (Departure Alert) ---
            elif getattr(self, 'departure', 0):
                # Świecimy wszystko na np. niebiesko lub cyjan
                alert_blue = (0, 0, 255)
                new_states = [alert_blue] * 8

            # --- POZIOM 3: STANDARDOWY STATUS (Gdy brak zagrożeń) ---
            else:
                # 0: Lewy kierunkowskaz
                if self.turnl:
                    new_states[0] = self.yellow
                
                # 2: Lewa linia
                if self.leftlane:
                    new_states[2] = self.green
                
                # 4: Wykrycie pojazdu (śledzenie - bezpieczne)
                if self.carlock:
                    new_states[4] = self.blue
                
                # 3: Prawa linia
                if self.rightlane:
                    new_states[3] = self.green
                
                # 5: Wykrycie światła (status, nie alarm)
                if self.light:
                    new_states[5] = self.red
                
                # 7: Prawy kierunkowskaz
                if self.turnr:
                    new_states[7] = self.yellow

            # --- Aktualizacja sprzętowa ---
            if new_states != self.led_states:
                self.led_states = new_states
                for i, color in enumerate(self.led_states):
                    # Rzutowanie na int dla bezpieczeństwa (biblioteka rpi_ws281x tego wymaga)
                    r, g, b = color
                    self.strip._pixels[i] = Color(int(r), int(g), int(b))
                self.strip.show()
                self.changed = False

        def show(self):
            """Kompatybilnosc wsteczna"""
            self.update_leds()

            # Reset flags
            self.turnl = 0
            self.leftlane = 0
            self.carlock = 0
            self.rightlane = 0
            self.light = 0
            self.turnr = 0

        def cleanup(self):
            self.clear_leds()

    class Beep:
        beep = [1, 440]
        beepbeat = [4, 4]
        playing = False

        def __init__(self):
            global buzzerPin
            # GPIO.setmode(GPIO.BCM)
            # GPIO.setup(buzzerPin, GPIO.OUT)
            # GPIO.setwarnings(False)
            # self.Buzz = GPIO.PWM(buzzerPin, 440)
            # self.Buzz.start(50)

        def playbeep(self, intensity):
            self.playing = True
            self.Buzz.start(5)
            for i in range(1, len(self.beep)):
                self.Buzz.ChangeFrequency(self.beep[i] * intensity)
                time.sleep(self.beepbeat[i] / intensity * 0.13)
            self.playing = False
            self.Buzz.stop()

        def play(self, intensity=1):
            if intensity > 7 or intensity < 1:
                return
            if not self.playing:
                Thread(target=self.playbeep, args=(intensity,)).start()
            else:
                print("already playing")


HISTORY_FILE = "obd_history.json"

def bluetooth_selection_process(devices, queue):
    """Funkcja uruchamiana w osobnym procesie, aby uniknąć konfliktów z wątkami ADAS"""
    import tkinter as tk
    from tkinter import ttk

    class OBDManager:
        def __init__(self, devices):
            self.root = tk.Tk()
            self.root.title("Wybór urządzenia OBDII")
            self.root.geometry("400x550")
            self.result = None
            
            tk.Label(self.root, text="Znalezione urządzenia Bluetooth:", font=('Arial', 10, 'bold')).pack(pady=10)
            
            self.tree = ttk.Treeview(self.root, columns=("MAC", "Nazwa"), show='headings', height=10)
            self.tree.heading("MAC", text="Adres MAC")
            self.tree.heading("Nazwa", text="Nazwa")
            self.tree.column("MAC", width=150)
            self.tree.column("Nazwa", width=200)
            self.tree.pack(pady=5, padx=10)

            for dev in devices:
                self.tree.insert("", tk.END, values=(dev['mac'], dev['name']))

            tk.Label(self.root, text="Kod parowania (PIN):").pack(pady=5)
            self.pin_entry = tk.Entry(self.root, justify='center')
            self.pin_entry.insert(0, "1234")
            self.pin_entry.pack()

            btn_frame = tk.Frame(self.root)
            btn_frame.pack(pady=20)
            
            tk.Button(btn_frame, text="Połącz z wybranym", command=self.on_connect, 
                      bg="#2ecc71", fg="white", width=20, height=2).pack(pady=5)
            
            tk.Button(btn_frame, text="Nie łącz aktualnie", command=self.on_skip, 
                      bg="#e74c3c", fg="white", width=20).pack(pady=5)

        def on_connect(self):
            selected = self.tree.selection()
            if selected:
                item = self.tree.item(selected)
                self.result = {
                    "mac": item['values'][0],
                    "name": item['values'][1],
                    "pin": self.pin_entry.get()
                }
                self.root.destroy()

        def on_skip(self):
            self.result = None
            self.root.destroy()

        def run(self):
            self.root.mainloop()
            return self.result

    manager = OBDManager(devices)
    selected_device = manager.run()
    queue.put(selected_device)

class OBDII:
    def __init__(self):
        self.cmd = obd.commands.SPEED
        self.oldSpeed = 0
        self.prevTime = datetime.datetime.now()
        self.currTime = datetime.datetime.now()
        self.accelLog = []
        self.avgAccel = 0
        self.connection = None
        self.is_connected = False
        self.port = None

    def _get_history(self):
        try:
            with open("obd_history.json", 'r') as f:
                return json.load(f)
        except:
            return None

    def _save_history(self, mac, name):
        with open("obd_history.json", 'w') as f:
            json.dump({"mac": mac, "name": name}, f)

    def _scan_bluetooth(self):
        print("Skanowanie Bluetooth...")
        devices = []
        try:
            subprocess.run(["bluetoothctl", "scan", "on"], timeout=4, stdout=subprocess.DEVNULL)
            out = subprocess.check_output(["bluetoothctl", "devices"]).decode()
            for line in out.splitlines():
                parts = line.split(" ", 2)
                if len(parts) >= 3:
                    devices.append({"mac": parts[1], "name": parts[2]})
        except: pass
        return devices

    def connectOBD(self):
        # 1. Historia
        history = self._get_history()
        if history:
            self.port = "/dev/rfcomm0"
            if self._attempt_connection():
                return True

        # 2. Wybór w osobnym procesie (izolacja Tkintera)
        bt_devices = self._scan_bluetooth()
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=bluetooth_selection_process, args=(bt_devices, queue))
        p.start()
        selected_device = queue.get() # Czeka na wynik z okna
        p.join() # Zamyka proces okna całkowicie

        if selected_device:
            try:
                mac = selected_device['mac']
                pin = selected_device['pin']
                print(f"Parowanie z {mac}...")
                subprocess.run(f'echo "pair {mac}\nexpect \"Enter PIN code:\"\nsend \"{pin}\r\"\ntrust {mac}\nquit" | bluetoothctl', shell=True)
                subprocess.run(["sudo", "rfcomm", "release", "0"], stderr=subprocess.DEVNULL)
                subprocess.run(["sudo", "rfcomm", "bind", "0", mac])
                time.sleep(2)
                self.port = "/dev/rfcomm0"
                if self._attempt_connection():
                    self._save_history(mac, selected_device['name'])
                    return True
            except Exception as e:
                print(f"Błąd: {e}")
        
        return False

    def _attempt_connection(self):
        try:
            self.connection = obd.Async(self.port, fast=False)
            self.connection.watch(self.cmd)
            self.connection.start()
            time.sleep(1)
            if self.connection.is_connected():
                self.is_connected = True
                return True
        except: pass
        return False

    def UpdateAccelSpeed(self):
        if not self.is_connected or not self.connection.is_connected():
            return
        global carSpeed, carAccel

        try:
            if self.is_connected and self.connection.is_connected():
                r = self.connection.query(self.cmd)
                if r.is_null():
                    return

                self.currTime = datetime.datetime.now()
                time = (self.currTime - self.prevTime).microseconds / 1000000
                if time == 0:
                    time = 0.001  # Unikaj dzielenia przez zero
                self.prevTime = self.currTime

                carSpeed = r.value.magnitude
                self.speedlog.append(carSpeed)

                accel = (r.value.magnitude - self.oldSpeed) / time
                self.oldSpeed = r.value.magnitude

                self.accelLog.append(accel)
                if len(self.accelLog) > accelN:
                    self.accelLog.pop(0)

                self.avgAccel = np.mean(self.accelLog)
                self.avglog.append(self.avgAccel)
                carAccel = self.avgAccel

                print(f"Speed: {carSpeed} km/h, AvgAccel: {self.avgAccel} m/s")
        except Exception as e:
            print(f"Błąd aktualizacji danych OBD: {e}")

    def cleanup(self):
        try:
            if self.connection:
                self.connection.stop()
                print("Połączenie OBD zamknięte")
        except Exception as e:
            print(f"Błąd zamykania połączenia OBD: {e}")


def parseArgs():
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modeldir",
        help="Folder the .tflite file is located in",
        default="TFLite_model",
    )  # ,required=True
    parser.add_argument(
        "--graph",
        help="Name of the .tflite file, if different than detect.tflite",
        default="model.tflite",
    )
    parser.add_argument(
        "--labels",
        help="Name of the labelmap file, if different than labelmap.txt",
        default="labelmap.txt",
    )
    parser.add_argument(
        "--threshold",
        help="Minimum confidence threshold for displaying detected objects",
        default=0.55,
    )
    parser.add_argument(
        "--resolution",
        help="Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.",
        default="720x720",
    )
    parser.add_argument(
        "--edgetpu",
        help="Use Coral Edge TPU Accelerator to speed up detection",
        action="store_true",
    )
    parser.add_argument(
        "--output_path", help="Where to save processed imges from pi.", default="images"
    )  # required=True

    args = parser.parse_args()
    return args


class VideoStream:
    def __init__(self, filename="", resolution=(800, 600), framerate=30):
        self.stopped = False
        self.frame = None
        self.ret = False
        self.resolution = resolution

        try:
            if not videoFileMode:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": resolution, "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(2)  # Daj kamerze czas na inicjalizacje
                self.ret = True
                self.frame = self.picam2.capture_array()
                print("Kamera zainicjalizowana pomyślnie")
            else:
                self.stream = cv2.VideoCapture(filename)
                if self.stream.isOpened():
                    self.ret = True
                    (self.ret, self.frame) = self.stream.read()
                    print(f"Plik wideo otwarty pomyślnie: {filename}")
                else:
                    print(f"Nie można otworzyć pliku wideo: {filename}")
                    self.frame = np.zeros(
                        (resolution[1], resolution[0], 3), dtype=np.uint8
                    )
        except Exception as e:
            print(f"Błąd inicjalizacji strumienia wideo: {e}")
            self.frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

        self.framerate = framerate

    def start(self):
        from threading import Thread

        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                if hasattr(self, "picam2"):
                    self.frame = self.picam2.capture_array()
                    self.ret = True
                elif hasattr(self, "stream"):
                    self.ret, self.frame = self.stream.read()
                    if not self.ret:
                        print("Koniec strumienia wideo")
                        self.stop()
                time.sleep(1 / self.framerate)
            except Exception as e:
                print(f"Błąd odczytu z kamery: {e}")
                self.ret = False
                time.sleep(0.1)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        try:
            if hasattr(self, "picam2"):
                self.picam2.stop()
            elif hasattr(self, "stream"):
                self.stream.release()
        except Exception as e:
            print(f"Błąd zatrzymywania strumienia: {e}")

    def get(self, t):
        if t == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frame.shape[1] if self.frame is not None else 0
        if t == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frame.shape[0] if self.frame is not None else 0
        return 0


# Lane Deprature class
class LaneDeparture:
    height = 0
    width = 0
    horizon = 0
    stathorizon = 0
    lanecenter = 0
    color_MIN = np.array([0, 0, 150], np.uint8)
    color_MAX = np.array([179, 255, 255], np.uint8)
    timerl = 0
    timerr = 0
    LLFrames = 0
    RLFrames = 0
    reachedleft = False
    reachedright = False
    averaged_lines = [[[0, 0, 0, 0]], [[0, 0, 0, 0]]]
    heightcrp = 0
    laneframe = []
    glassGradient = []

    def __init__(self, cap):
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.heightcrp = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.heightcrp > self.width:
            self.height = int(self.width * 9 / 16)
        else:
            self.height = self.heightcrp - hood

        self.horizon = self.height * starting_horizon_Ratio
        self.stathorizon = self.horizon
        self.lanecenter = self.width / 2
        self.glassGradient = self.createGlassGradient()

    def get_gradient_2d(self, start, stop, width, height, is_horizontal):
        if is_horizontal:
            return np.tile(np.linspace(start, stop, width), (height, 1))
        else:
            return np.tile(np.linspace(start, stop, height), (width, 1)).T

    def get_gradient_3d(self, width, height, start_list, stop_list, is_horizontal_list):
        result = np.zeros((height, width, len(start_list)), dtype=np.uint8)

        for i, (start, stop, is_horizontal) in enumerate(
            zip(start_list, stop_list, is_horizontal_list)
        ):
            result[:, :, i] = self.get_gradient_2d(
                start, stop, width, height, is_horizontal
            )

        return result

    def createGlassGradient(self):
        global glassGradStreght

        array = self.get_gradient_3d(
            self.width,
            self.heightcrp,
            (0, 0, 0),
            (glassGradStreght, glassGradStreght, glassGradStreght),
            (False, False, False),
        )

        return array

    def make_points(self, image, line):
        slope, intercept = line
        # print(slope, intercept)
        y1 = int(self.height)  # bottom of the image
        y2 = int(
            self.horizon + self.horizon * 15 / 100
        )  # slightly lower than the middle
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [[x1, y1, x2, y2]]

    def average_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []
        Rlane = image.shape[1] / 2 + 50
        Llane = image.shape[1] / 2 - 50
        # global self.timerl
        # global self.timerr
        # global averaged_lines
        # global avgSlope
        # temp = averaged_lines
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y1 - y2) / (x1 - x2)
                    if 7 > slope > 0.3:
                        if x1 > Rlane:
                            yintercept = y2 - (slope * x2)
                            right_fit.append((slope, yintercept))
                    elif -7 < slope < -0.3:
                        if x1 < Llane:
                            yintercept = y2 - (slope * x2)
                            left_fit.append((slope, yintercept))
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = self.make_points(image, left_fit_average)
            self.timerl = 0
        else:
            if self.timerl <= memoryframes:
                left_line = self.averaged_lines[0]
                self.timerl += 1
            else:
                left_line = [[0, 0, 0, 0]]
                self.timerl = 0
        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = self.make_points(image, right_fit_average)
            self.timerr = 0
        else:
            if self.timerr <= memoryframes:
                right_line = self.averaged_lines[1]
                self.timerr += 1
            else:
                right_line = [[0, 0, 0, 0]]
                self.timerr = 0

        oldLineW = 1 - newLineW
        if any(left_line[0]):
            if any(self.averaged_lines[0][0]):
                self.averaged_lines[0][0][0] = int(
                    self.averaged_lines[0][0][0] * oldLineW + left_line[0][0] * newLineW
                )
                self.averaged_lines[0][0][1] = int(
                    self.averaged_lines[0][0][1] * oldLineW + left_line[0][1] * newLineW
                )
                self.averaged_lines[0][0][2] = int(
                    self.averaged_lines[0][0][2] * oldLineW + left_line[0][2] * newLineW
                )
                self.averaged_lines[0][0][3] = int(
                    self.averaged_lines[0][0][3] * oldLineW + left_line[0][3] * newLineW
                )
            else:
                self.averaged_lines[0] = left_line
        else:
            self.averaged_lines[0] = left_line
        if any(right_line[0]):
            if any(self.averaged_lines[1][0]):
                self.averaged_lines[1][0][0] = int(
                    self.averaged_lines[1][0][0] * oldLineW
                    + right_line[0][0] * newLineW
                )
                self.averaged_lines[1][0][1] = int(
                    self.averaged_lines[1][0][1] * oldLineW
                    + right_line[0][1] * newLineW
                )
                self.averaged_lines[1][0][2] = int(
                    self.averaged_lines[1][0][2] * oldLineW
                    + right_line[0][2] * newLineW
                )
                self.averaged_lines[1][0][3] = int(
                    self.averaged_lines[1][0][3] * oldLineW
                    + right_line[0][3] * newLineW
                )
            else:
                self.averaged_lines[1] = right_line
        else:
            self.averaged_lines[1] = right_line
        return self.averaged_lines

    def canny(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur1 = cv2.GaussianBlur(gray, (5, 5), 0, 0)
        # cv2.imshow("blur",blur1)
        canny = cv2.Canny(blur1, 80, 140)
        return canny

    def get_intersect(self, averaged_lines):
        if averaged_lines == None:
            return 0, 0

        # Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        # a1: [x, y] a point on the first line
        a1 = [averaged_lines[0][0][0], averaged_lines[0][0][1]]
        # a2: [x, y] another point on the first line
        a2 = [averaged_lines[0][0][2], averaged_lines[0][0][3]]
        # b1: [x, y] a point on the second line
        b1 = [averaged_lines[1][0][0], averaged_lines[1][0][1]]
        # b2: [x, y] another point on the second line
        b2 = [averaged_lines[1][0][2], averaged_lines[1][0][3]]

        s = np.vstack([a1, a2, b1, b2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            # lanecenterleft = np.interp(self.horizon, a1, a2)
            # s for stacked
            sl = np.vstack([a1, a2, [0, self.horizon], [self.width, self.horizon]])
            hl = np.hstack((sl, np.ones((4, 1))))  # h for homogeneous
            ll1 = np.cross(hl[0], hl[1])  # get first line
            hl2 = np.cross(hl[2], hl[3])  # get second line
            xl, yl, zl = np.cross(ll1, hl2)
            # lanecenterright = np.interp(self.horizon, b1, b2)
            # s for stacked
            sr = np.vstack([b1, b2, [0, self.horizon], [self.width, self.horizon]])
            hr = np.hstack((sr, np.ones((4, 1))))  # h for homogeneous
            rl1 = np.cross(hr[0], hr[1])  # get first line
            hl2 = np.cross(hr[2], hr[3])  # get second line
            xr, yr, zr = np.cross(rl1, hl2)
            if zl != 0:
                # print("lanecenterleft")
                return int((self.horizon + self.stathorizon) / 2), int(xl / zl)
            elif zr != 0:
                # print("lanecenterright")
                return int((self.horizon + self.stathorizon) / 2), int(xr / zr)
            else:
                # print("reset")
                return int((self.horizon + self.stathorizon) / 2), int(
                    (self.lanecenter + self.width / 2) / 2
                )
        return int(y / z), int(x / z)

    def checklaneDeparture(self, img, averaged_lines):
        # global self.LLFrames, self.reachedleft
        # global self.RLFrames, self.reachedright
        # global self.lanecenter,self.height,self.width,self.horizon
        mask = np.zeros_like(img)
        carPathSpace = np.array(
            [
                [
                    (self.width / 2 - carsize / 2 + camOffset, self.height + wheelOff),
                    (self.lanecenter, self.horizon),
                    (self.lanecenter, self.horizon),
                    (self.width / 2 + carsize / 2 + camOffset, self.height + wheelOff),
                ]
            ],
            np.int32,
        )

        mask_limit = np.array(
            [
                [
                    (0, 0),
                    (0, self.horizon * lane_depth),
                    (self.width, self.horizon * lane_depth),
                    (self.width, 0),
                ]
            ],
            np.int32,
        )

        cv2.fillPoly(mask, carPathSpace, color=(0, 0, 255))
        cv2.fillPoly(mask, mask_limit, 0)
        # cv2.imshow("carPathSpace", mask)
        # print(averaged_lines)
        a1 = (averaged_lines[0][0][0], averaged_lines[0][0][1])
        resultLeftBot = cv2.pointPolygonTest(carPathSpace, a1, False)
        b1 = (averaged_lines[1][0][0], averaged_lines[1][0][1])
        resultRightBot = cv2.pointPolygonTest(carPathSpace, b1, False)

        if averaged_lines[0] != [[0, 0, 0, 0]]:  # if in the frame a light is found
            # if previous x(thresh) frames had the same light
            if self.LLFrames > laneLockThresh:
                self.reachedleft = True
            else:  # else add it to the list
                self.LLFrames = self.LLFrames + 1
        else:  # if there is no lane in the frame
            if self.LLFrames > 0:  # if previous frames had lane
                self.LLFrames = self.LLFrames - 1  # remove a frame from history
        if self.LLFrames == 0:
            self.reachedleft = False

        if averaged_lines[1] != [[0, 0, 0, 0]]:
            if self.RLFrames > laneLockThresh:
                self.reachedright = True
            else:  # else add it to the list
                self.RLFrames = self.RLFrames + 1
        else:  # if there is no lane in the frame
            # print('no right')
            if self.RLFrames > 0:  # if previous frames had lane
                self.RLFrames = self.RLFrames - 1  # remove a frame from history
        if self.RLFrames == 0:
            self.reachedright = False

        if self.reachedright:
            if RaspberryPi:
                statled.set_state(rightlane=1)
            mask = warn(resultRightBot, mask)
            cv2.putText(
                mask,
                "R",
                (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
        if self.reachedleft:
            if RaspberryPi:
                statled.set_state(leftlane=1)
            mask = warn(resultLeftBot, mask)
            cv2.putText(
                mask,
                "L",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
        if averaged_lines is None:
            return np.zeros_like(img)
        else:
            return mask

    def display_lines(self, img, lines):
        # global self.lanecenter,self.height,self.width,self.horizon,self.stathorizon
        line_image = np.zeros_like(img)
        self.width = img.shape[1]
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    # if  0 <= x1 <= 3000 and 0 <= y1 <= 3000 and 0 <= x2 <= 3000 and 0 <= y2 <= 3000:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        cv2.line(
            line_image,
            (1, int(self.horizon)),
            (self.width, int(self.horizon)),
            (0, 255, 0),
            1,
        )
        cv2.line(
            line_image,
            (1, int(self.stathorizon)),
            (self.width, int(self.stathorizon)),
            (0, 255, 20),
            1,
        )
        cv2.line(
            line_image,
            (int(self.lanecenter), 1),
            (int(self.lanecenter), int(self.height)),
            (0, 255, 0),
            1,
        )
        return line_image

    def region_of_interest(self, canny):
        # global self.lanecenter,self.height,self.width,self.horizon,roi

        self.width = canny.shape[1]
        mask = np.zeros_like(canny)

        roi = np.array(
            [
                [
                    (fov + camOffset, self.height - hood),
                    (self.lanecenter, self.horizon),
                    (self.width / 2 - carsize / 7 + camOffset, self.height - hood),
                    (self.width / 2 + carsize / 7 + camOffset, self.height - hood),
                    (self.lanecenter, self.horizon),
                    (self.width - fov + camOffset, self.height - hood),
                ]
            ],
            np.int32,
        )

        mask_limit = np.array(
            [
                [
                    (0, 0),
                    (0, self.horizon * lane_depth),
                    (self.width, self.horizon * lane_depth),
                    (self.width, 0),
                ]
            ],
            np.int32,
        )

        cv2.fillPoly(mask, roi, 255)
        cv2.fillPoly(mask, mask_limit, 0)
        # mask = cv2.bitwise_and(mask, mask, mask=mask_limit)
        # cv2.imshow("mask", mask)
        masked_image = cv2.bitwise_and(canny, mask)
        return masked_image, mask

    def get_road_brightness(self, hsv_img):
        global carsize, hood, brightsens

        ymin = int(self.horizon + (self.height - self.horizon) * 50 / 100) - 1
        ymax = int(self.height - hood) + 1
        xmin = int(self.lanecenter - carsize / 5) - 1
        xmax = int(self.lanecenter + carsize / 5) + 1
        masked_image = hsv_img[ymin:ymax, xmin:xmax]

        gray = cv2.cvtColor(hsv_img, cv2.COLOR_RGB2GRAY)
        brightmask = np.zeros_like(gray)

        mask = np.array(
            [
                [
                    (xmin, ymin),
                    (xmax, ymin),
                    (
                        xmax,
                        ymax,
                    ),
                    (
                        xmin,
                        ymax,
                    ),
                ]
            ],
            np.int32,
        )

        cv2.fillPoly(brightmask, mask, 255)
        # cv2.imshow("mask", masked_image)
        # find max per row
        average_color_per_row = np.max(masked_image, axis=1, initial=1)
        # find average across max per row
        average_color = np.average(average_color_per_row, axis=0)
        # convert back to uint8
        average_color = np.uint8(average_color)

        road_bright = average_color[2]
        return road_bright + brightsens, brightmask

    def laneDeparture(self, frame):
        # global self.lanecenter,self.height,self.width,self.stathorizon,self.horizon,averaged_lines,roi

        if self.heightcrp > self.width:
            frame = frame[
                int(self.heightcrp / 2 - self.height / 2 - hood) : int(
                    self.heightcrp / 2 + self.height / 2 - hood
                ),
            ]
        frame = cv2.addWeighted(frame, 0.8, self.glassGradient, 1, 1)
        # cv2.imshow("frame", frame)
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roadbright, brightmask = self.get_road_brightness(hsv_img)
        self.color_MIN = np.array(
            [
                0,
                0,
                self.color_MIN[2] * (1 - smothBrightfact)
                + smothBrightfact * roadbright,
            ],
            np.uint8,
        )
        frame_threshedW = cv2.inRange(hsv_img, self.color_MIN, self.color_MAX)
        resultWfr = cv2.bitwise_and(frame, frame, mask=frame_threshedW)
        canny_image = self.canny(resultWfr)
        cropped, roi = self.region_of_interest(canny_image)
        lines = cv2.HoughLinesP(
            cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=60, maxLineGap=5
        )
        self.average_slope_intercept(frame, lines)
        departure = self.checklaneDeparture(frame, self.averaged_lines)
        newhorizon, newlanecenter = self.get_intersect(self.averaged_lines)
        if newhorizon != 0:  # and 0<self.lanecenter<self.width:
            self.horizon = newhorizon
            self.stathorizon = self.stathorizon * 0.99 + self.horizon * 0.01
            self.lanecenter = (
                self.lanecenter * (1 - smothLaneCenter)
                + newlanecenter * smothLaneCenter
            )
        else:
            self.stathorizon = (
                self.stathorizon * 0.99 + self.height * starting_horizon_Ratio * 0.01
            )
        print("done lane")

        if debugDisplay:
            line_image = self.display_lines(frame, self.averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, departure, 1, 1)
            combo_image = cv2.addWeighted(
                combo_image,
                1,
                cv2.bitwise_and(combo_image, combo_image, mask=cropped),
                1,
                0.1,
            )
            combo_image = cv2.addWeighted(
                combo_image,
                1,
                cv2.bitwise_and(combo_image, combo_image, mask=roi),
                0.3,
                1,
            )
            combo_image = cv2.addWeighted(
                combo_image,
                1,
                cv2.bitwise_and(combo_image, combo_image, mask=brightmask),
                0.3,
                1,
            )
            combo_image = cv2.addWeighted(combo_image, 1, line_image, 1, 1)

            self.laneframe = combo_image


# Infrancing Model class
class Model:

    def __init__(self):
        args = parseArgs()
        MODEL_NAME = args.modeldir
        GRAPH_NAME = args.graph
        LABELMAP_NAME = args.labels
        self.min_conf_threshold = float(args.threshold)
        # resW, resH = args.resolution.split('x')
        self.imW, self.imH = vheight, vheight
        use_TPU = args.edgetpu

        # Import TensorFlow libraries
        # If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec("tensorflow")
        if pkg is None:
            from tflite_runtime.interpreter import Interpreter

            if use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter

            if use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if GRAPH_NAME == "detect.tflite":
                GRAPH_NAME = "edgetpu.tflite"

            # Get path to current working directory
        CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = "/home/edex/Desktop/Pi-ADAS/TFLite_model/model.tflite"

        # Path to label map file
        PATH_TO_LABELS = "/home/edex/Desktop/Pi-ADAS/TFLite_model/labelmap.txt"

        # Load the label map
        with open(PATH_TO_LABELS, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.labels[0] == "???":
            del self.abels[0]
        print(self.labels)
        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if use_TPU:
            self.interpreter = Interpreter(
                model_path=PATH_TO_CKPT,
                experimental_delegates=[load_delegate("libedgetpu.so.1.0")],
            )
            print(PATH_TO_CKPT)
        else:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT)

        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]["shape"][1]
        self.width = self.input_details[0]["shape"][2]

        # print(output_details)

        # led_on = False
        self.floating_model = self.input_details[0]["dtype"] == np.float32

        input_mean = 127.5
        input_std = 127.5

    def detectObj(self, frame):
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))

        input_data = np.expand_dims(frame_resized, axis=0)
        # cv2.imshow('model view', frame_resized)
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        # Model.interpreter.get_input_details()
        # Model.interpreter.get_output_details()

        self.interpreter.invoke()

        # Retrieve detection results
        # Bounding box coordinates of detected objects
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        # Class index of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        # Confidence of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        # Total number of detected objects (inaccurate and not needed)
        # num = Model.interpreter.get_tensor(Model.output_details[3]['index'])[0]
        return boxes, classes, scores


# Object Colision class
class objColision:
    oldCOIcenter = 0
    oldCOIcentery = 0
    oldCOIylenght = 0
    oldCOIxlenght = 0
    oldCOIarea = 0
    oldCOIareaDep = 0
    Dep = False
    frDep = 0
    cancel = 0

    #  Object tracking Variables
    # params for ShiTomasi corner detection
    maxcorners = 4
    feature_params = dict(
        maxCorners=maxcorners, qualityLevel=0.3, minDistance=7, blockSize=7
    )
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(10, 10),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    findNew = False
    first = True
    old_gray = []
    p0 = 0
    dist = 0
    oldDist = 0
    trackdistChange = 0
    checkNew = False
    oldCenter = 0

    def initFrameVar(self):
        self.COIymax = 0
        self.COIxmin = 0
        self.COIymin = 0
        self.COIxmax = 0
        self.COIcenter = 0
        self.COIcentery = 0
        self.COIylenght = 0
        self.COIxlenght = 0
        self.COIarea = 0
        self.inPath = False
        self.newCar = False

    def objTracking(self):
        global frame, frameout, warned

        pad = 20
        ratio = 1
        # cv2.imshow("trackimage",frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        maskzero = np.zeros_like(frame_gray)
        if self.COIxmin == 0:
            pad = 0
        else:
            ratio = 40 / self.COIylenght
        maskratio = pad / ratio / 2
        outratio = ratio * pad
        maskb = np.array(
            [
                [
                    (self.COIxmin + maskratio, self.COIymin + maskratio),
                    (self.COIxmax - maskratio, self.COIymin + maskratio),
                    (
                        self.COIxmax - maskratio,
                        self.COIymax - maskratio,
                    ),
                    (
                        self.COIxmin + maskratio,
                        self.COIymax - maskratio,
                    ),
                ]
            ],
            np.int32,
        )
        outofbounds = np.array(
            [
                [
                    (self.COIxmin - outratio, self.COIymin - outratio),
                    (self.COIxmax + outratio, self.COIymin - outratio),
                    (
                        self.COIxmax + outratio,
                        self.COIymax + outratio,
                    ),
                    (
                        self.COIxmin - outratio,
                        self.COIymax + outratio,
                    ),
                ]
            ],
            np.int32,
        )
        box = cv2.fillPoly(maskzero, maskb, 255)
        mask = np.zeros_like(frame)
        if self.first:
            self.old_gray = frame_gray
            self.p0 = cv2.goodFeaturesToTrack(
                frame_gray, mask=box, **self.feature_params
            )
            # Create a mask image for drawing purposes
            self.first = False

        if self.findNew and self.COIxmin != 0 and self.COIylenght > 50:
            self.p0 = cv2.goodFeaturesToTrack(
                self.old_gray, mask=box, **self.feature_params
            )
            mask = np.zeros_like(frame)
            self.dist = 0
            self.oldDist = 0
            self.trackdistChange = 0
            self.oldCenter = 0
            self.findNew = False
        if self.p0 is None:
            self.findNew = True
            return

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray, frame_gray, self.p0, None, **self.lk_params
        )
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            if len(good_new) < self.maxcorners:
                self.findNew = True
                return
            good_old = self.p0[st == 1]
            # draw the tracks
            centerx = []
            centery = []
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()

                if self.checkNew and self.COIxmin != 0:
                    re = cv2.pointPolygonTest(outofbounds, (a, b), False)
                    self.checkNew = False
                else:
                    re = 1

                centerx.append(a)
                centery.append(b)
                c, d = old.ravel()
                mask = cv2.line(
                    mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 4
                )
                frameout = cv2.circle(
                    frame, (int(a), int(b)), 15, self.color[i].tolist(), -1
                )

                if (
                    vheight - 40 <= a
                    or a <= 40
                    or vwidth - 40 <= b
                    or b <= 40
                    or re == -1
                ):
                    self.findNew = True
                    break
            # for i in range(0,len(good_new)-1):
            if len(good_new) > self.maxcorners - 1 and not self.findNew:
                center = [np.mean(centerx), np.mean(centery)]
                offset = 0
                if self.oldCenter != 0:
                    offset = abs(self.oldCenter[0] - center[0])
                self.oldCenter = center
                print(offset)
                if offset > trackingoffThresh:
                    # warned=True
                    self.findNew = True
                if not self.findNew:
                    frameout = cv2.circle(
                        frame,
                        (int(center[0]), int(center[1])),
                        5,
                        self.color[5].tolist(),
                        -1,
                    )
                    dist = []
                    for i in range(0, len(good_new) - 1):
                        dist.append(math.dist(good_new[i].ravel(), center))

                    self.dist = np.mean(dist)
                    if self.oldDist != 0:
                        self.trackdistChange = (self.oldDist - self.dist) / self.oldDist
                    if self.trackdistChange > 0.1:
                        self.findNew = True
                    self.oldDist = self.dist
        else:
            self.findNew = True
        # cv2.rectangle(frameout, (int(self.COIxmin+maskratio), int(self.COIymin+maskratio)), (int(self.COIxmax-maskratio), int(self.COIymax-maskratio)), (255, 0, 255), 2)
        # cv2.rectangle(frameout, (int(self.COIxmin-outratio), int(self.COIymin-outratio)), (    int(self.COIxmax+outratio), int(self.COIymax+outratio)), (255, 0, 255), 2)
        frameout = cv2.add(frameout, mask)
        cv2.putText(
            frameout,
            str("%.2f" % self.trackdistChange),
            (10, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        # Now update the previous frame and previous points
        self.old_gray = frame_gray.copy()
        if p1 is not None:
            self.p0 = good_new.reshape(-1, 1, 2)

    def drawObjCol(self, mode):
        global debugDisplay, frameout
        if debugDisplay:
            # mode 0 for frame draw and
            # mode 1 for time draw
            if mode == 0:
                if self.inPath:
                    cv2.rectangle(frameout, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frameout, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                label = "%s: %d%%" % (
                    object_name,
                    int(scores[i] * 100),
                )  # Example: 'person: 72%'
                # label = ''
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )  # Get font size
                label_ymin = max(
                    ymin, labelSize[1] + 10
                )  # Make sure not to draw label too close to top of window
                cv2.rectangle(
                    frameout,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    (255, 255, 255),
                    cv2.FILLED,
                )  # Draw white box to put label text in
                cv2.putText(
                    frameout,
                    label,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )  # Draw label text
            if mode == 1:
                if self.newCar:  # checked if new car in pt1
                    cv2.rectangle(
                        frameout,
                        (self.COIxmin, self.COIymin),
                        (self.COIxmax, self.COIymax),
                        (255, 0, 0),
                        2,
                    )
                else:
                    cv2.rectangle(
                        frameout,
                        (self.COIxmin, self.COIymin),
                        (self.COIxmax, self.COIymax),
                        (0, 255, 255),
                        2,
                    )

    def getObjOfIntrest(self):
        # global inPath,newCar,COIymax,COIxmin,COIymin,COIxmax,COIcenter,COIylenght,COIarea
        # To find the object that are in the straight path of the car
        # Calculate a weight for every car and vru based on the offset from the center but also the size of the object
        weight = (
            ylenght
            * weight_size
            / (abs(vwidth / 2 + camOffset - center) + 0.5)
            * weight_center
        )

        # if the weight larger than paththresh and
        if weight >= inPathSens:
            # mark them red
            self.inPath = True
            # Select closest car as car of intrest
            # (after every car has passed this function only the closest car will matter)
            # (a car that is closer is also southest in the 2d perspective projection plane)
            # based on the bottom of the object rectangle
            if ymax > self.COIymax:
                # check if the center of the car compared to the previous car of intrest is above thresh
                # if a new car gets in front slowly then the center of the coi will change
                # if it gets in front fast <1 frame then its probably better to warn driver anyway
                if (
                    abs(center - self.oldCOIcenter) > newObjThresh
                    or abs(centery - self.oldCOIcentery) > newObjThresh * 2
                ):
                    self.newCar = True  # then it is a new car
                else:
                    self.newCar = False
                # add all values as coi values
                self.COIymax = ymax
                self.COIxmin = xmin
                self.COIymin = ymin
                self.COIxmax = xmax
                self.COIcenter = center
                self.COIcentery = centery
                self.COIylenght = ylenght
                self.COIxlenght = xlenght
                self.COIarea = area
        else:
            self.inPath = False

    def objColWarn(self):
        global carSpeed, frame, frameout
        # Find if there is a collision or departure if stoped by mesuring the change in size of the car in front
        #       (coi) and comparing it to a threshhold but also taking into account if threre is a new car
        # if self.COIarea>2000:
        self.objTracking()
        if self.newCar:  # checked if new car in pt1
            # remove old car of intrest area and mark new car as new car of intrest
            self.oldCOIarea = 0
            self.oldCOIxlenght = 0
            self.oldCOIylenght = 0
            self.oldCOIareaDep = 0
            # self.Dep=False
            # self.checkNew=True
        else:
            if RaspberryPi:
                statled.carlock = 1
            # car locked as car of intrest
            # check if the positive change in size of the car of intrest is above the thresh but also not a new car for only one frame

            self.checkNew = True

            if self.oldCOIylenght != 0 and self.oldCOIxlenght != 0:  # and carSpeed >= 5
                yChangeP = (self.COIylenght - self.oldCOIylenght) / self.oldCOIylenght
                xChangeP = (self.COIxlenght - self.oldCOIxlenght) / self.oldCOIxlenght
                # areachange = self.COIarea - self.oldCOIarea
                # cv2.putText(frameout, str("%.2f" % yChangeP), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                #                 cv2.LINE_AA)
                # cv2.putText(frameout, str("%.2f" % xChangeP), (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                #                 cv2.LINE_AA)
                if (
                    self.COIxlenght > 150
                    and (yChangeP > CollisionSens or xChangeP > CollisionSens)
                    and self.trackdistChange < -trackingSens
                ):  # areachange > CollisionThresh and
                    print("COLLISION ALERT")
                    if RaspberryPi:
                        statled.set_state(collision=1) # WŁĄCZ ALARM
                        statled.update_leds()
                    frameout = warn(2, frameout)
                else:
                    if RaspberryPi and hasattr(statled, 'collision') and statled.collision:
                        statled.set_state(collision=0) # WYŁĄCZ ALARM
                        statled.update_leds()
            # check if the negative change in size of the car of intrest is above the thresh but also not a new car for only one frame
            self.oldCOIarea = self.COIarea  # make new values old
            self.oldCOIxlenght = self.COIxlenght
            self.oldCOIylenght = self.COIylenght

            if self.Dep:
                # cv2.putText(frame, 'dep checking', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2,
                #                cv2.LINE_AA)
                cv2.putText(
                    frameout,
                    str(self.frDep),
                    (120, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
                if self.trackdistChange > 0.015 and carSpeed <= 5:
                    if self.frDep >= departureSens:
                        print("Departure Alert")
                        frameout = warn(3, frameout)
                        self.Dep = False
                    self.oldCOIareaDep = self.COIarea
                    self.oldCOIarea = self.COIarea
                    self.frDep = self.frDep + 1
                    self.cancel = 0
                else:
                    # cv2.putText(frameout, str(self.cancel), (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,cv2.LINE_AA)
                    self.cancel = self.cancel + 1
                    if self.cancel > 5:
                        self.Dep = False
                        self.cancel = 0

            if (
                self.oldCOIarea != 0
                and self.COIarea != 0
                and self.COIarea > departureThresh
                and carSpeed <= 5
            ):  #
                if not self.Dep:
                    self.Dep = True
                    self.frDep = 0
                    # cv2.putText(frame, 'dep enabled', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                    #                 cv2.LINE_AA)
                    self.oldCOIareaDep = self.COIarea
                    self.oldCOIarea = self.COIarea

        # save the new center so we can check for new car in next frame
        self.oldCOIcenter = self.COIcenter
        self.oldCOIcentery = self.COIcentery


# Light Color Detection and Warning class
class LightDet:
    oldLOIcenter = 0
    oldLOIylenght = 0
    oldLOIarea = 0
    oldLOIcolor = 0
    reachedg = False
    reachedr = False
    redWarn = False
    warntime = 100
    lightxt = ""

    #  Object tracking Variables
    # params for ShiTomasi corner detection
    maxcorners = 1
    feature_params = dict(
        maxCorners=maxcorners, qualityLevel=0.1, minDistance=7, blockSize=7
    )
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(10, 10),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    findNew = False
    first = True
    old_gray = []
    p0 = 0
    checkNew = False

    def lightTracking(self):
        global frame, frameout, warned

        frameInv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv_imgInv = cv2.cvtColor(frameInv, cv2.COLOR_BGR2HSV)
        hsv_img_red = cv2.inRange(hsv_imgInv, color_MINr, color_MAXr)
        hsv_img_green = cv2.inRange(hsv_imgInv, color_MINg, color_MAXg)
        mask = cv2.bitwise_or(hsv_img_red, hsv_img_green)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        frame_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame_gray', frame_gray)
        maskzero = np.zeros_like(frame_gray)

        pad = 20
        # ratio = 1
        # if self.LOIxmin == 0:
        #     pad =0
        # else:
        #     ratio = 40/(self.LOIymax - self.LOIymin)
        maskratio = pad
        maskb = np.array(
            [
                [
                    (self.LOIxmin - maskratio, self.LOIymin - maskratio),
                    (self.LOIxmax + maskratio, self.LOIymin - maskratio),
                    (
                        self.LOIxmax + maskratio,
                        self.LOIymax + maskratio,
                    ),
                    (
                        self.LOIxmin - maskratio,
                        self.LOIymax + maskratio,
                    ),
                ]
            ],
            np.int32,
        )

        # cv2.rectangle(frameout, (int(self.LOIxmin-maskratio), int(self.LOIymin-maskratio)), (int(self.LOIxmax+maskratio), int(self.LOIymax+maskratio)), (255, 0, 255), 2)
        box = cv2.fillPoly(maskzero, maskb, 255)
        mask = np.zeros_like(frame)

        if self.first:
            self.old_gray = frame_gray
            self.p0 = cv2.goodFeaturesToTrack(
                frame_gray, mask=box, **self.feature_params
            )
            # Create a mask image for drawing purposes
            self.first = False

        if self.findNew and self.LOIxmin != 0:
            self.p0 = cv2.goodFeaturesToTrack(
                self.old_gray, mask=box, **self.feature_params
            )
            mask = np.zeros_like(frame)
            self.oldCenter = 0
            self.findNew = False

        if self.p0 is None:
            self.findNew = True
            return

        centerx = 0
        centery = 0
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray, frame_gray, self.p0, None, **self.lk_params
        )
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            # draw the tracks
            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            if len(good_new) > 0:
                a, b = good_new.ravel()

                if self.checkNew and self.LOIxmin != 0:
                    re = cv2.pointPolygonTest(maskb, (a, b), False)
                    self.checkNew = False
                else:
                    re = 1

                centerx = a
                centery = b

                c, d = good_old.ravel()
                mask = cv2.line(
                    mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 4
                )
                frameout = cv2.circle(
                    frame, (int(a), int(b)), 15, self.color[i].tolist(), -1
                )

                changex = a - c
                changey = b - d

                if vheight <= a or a <= 0 or vwidth <= b or b <= 0 or re == -1:
                    self.findNew = True
        else:
            # cv2.putText(frameout, "Not good enough point", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,cv2.LINE_AA)
            self.findNew = True

        if centerx and centery and self.LOIxmin:
            self.trackymin = self.LOIymin - maskratio
            self.trackymax = self.LOIymax + maskratio
            self.trackxmin = self.LOIxmin - maskratio
            self.trackxmax = self.LOIxmax + maskratio

            # self.trackxlenght = self.LOIxmax - self.LOIxmin
            # self.trackylenght = self.LOIymax - self.LOIymin
            # self.trackCenter = (a,b)
        elif centerx and centery:
            self.trackymin = self.trackymin - changey
            self.trackymax = self.trackymax - changey
            self.trackxmin = self.trackxmin - changex
            self.trackxmax = self.trackxmax - changex
        # cv2.rectangle(frameout, (int(self.trackxmin-maskratio), int(self.trackymin-maskratio)), (int(self.trackxmax+maskratio), int(self.trackymax+maskratio)), (255, 200, 255), 2)
        frameout = cv2.add(frameout, mask)
        # cv2.putText(frameout, str("%.2f" % self.trackdistChange), (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,cv2.LINE_AA)
        # Now update the previous frame and previous points
        self.old_gray = frame_gray.copy()
        if p1 is not None:
            self.p0 = good_new.reshape(-1, 1, 2)

    def __init__(self):
        self.warnStart = 0

    def initFrameVar(self):
        global vheight, vwidth
        self.isActualLight = False
        self.lightcolor = True
        self.pdoublelightcent = 0
        self.LOIs = False
        self.LOIymin = vheight
        self.LOIymax = vheight
        self.LOIxmin = 0
        self.LOIxmax = 0
        self.LOIcenterx = vwidth / 2
        self.LOIcentery = vheight
        self.LOIcolor = False
        self.doublelight = False
        self.prevdoublelight = False

    def getLightColorWindows(self, frame):
        try:
            crop_img = frame[
                abs(ymin - int(ylenght / 2)) : ymax + int(ylenght / 2),
                abs(xmin - xlenght) : xmax + xlenght,
            ]
            crop_imgInv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            hsv_imgInv = cv2.cvtColor(crop_imgInv, cv2.COLOR_BGR2HSV)
            hsv_img_red = cv2.inRange(hsv_imgInv, color_MINr, color_MAXr)
            hsv_img_green = cv2.inRange(hsv_imgInv, color_MINg, color_MAXg)

            # cv2.imshow('hsv_img_red', hsv_img_red)
            # cv2.imshow('hsv_img_green', hsv_img_green)

            return hsv_img_red, hsv_img_green
        finally:
            crop_img = np.zeros_like(crop_img)
            hsv_img_red = np.zeros_like(hsv_img_red)
            hsv_img_green = np.zeros_like(hsv_img_green)

        # hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        # hsv_img_green = cv2.inRange(hsv_img, color_MINg, color_MAXg)

        # numRed = cv2.countNonZero(hsv_img_red)
        # numGreen = cv2.countNonZero(hsv_img_green)

        # crop_img = np.zeros_like(crop_img)
        # hsv_img_red = np.zeros_like(hsv_img_red)
        # hsv_img_green = np.zeros_like(hsv_img_green)

    def checkDoubleLight(self, hsv_img_red, hsv_img_green):
        # global self.isActualLight, self.lightcolor, self.doublelight, lightxt,tsign_left,tsign_right
        doubleleft, doubleright = False, False
        if (
            cv2.countNonZero(hsv_img_red) > lightthresh
            and cv2.countNonZero(hsv_img_green) > lightthresh
        ):
            self.doublelight = True
            self.isActualLight = True
            # self.pdoublelightcent= center
            hsv_img_red_left = hsv_img_red[
                0 : hsv_img_red.shape[0], 0 : int(hsv_img_red.shape[1] / 2)
            ]
            hsv_img_red_right = hsv_img_red[
                0 : hsv_img_red.shape[0],
                int(hsv_img_red.shape[1] / 2) : hsv_img_red.shape[1],
            ]
            if cv2.countNonZero(hsv_img_red_left) > cv2.countNonZero(hsv_img_red_right):
                # print("left red straight green")
                self.lightxt = "<Red|Green^"
                doubleleft = True
                doubleright = False
            else:
                # print("right red straight green")
                self.lightxt = "^Green|Red>"
                doubleleft = False
                doubleright = True
        else:
            self.doublelight = False
            # self.isActualLight = False

        #  Check color of intrest with turn signal
        if tsign_left and doubleleft:
            self.lightcolor = False
        elif tsign_right and doubleright:
            self.lightcolor = False
        else:
            self.lightcolor = True

    def checkNormalight(self, hsv_img_red, hsv_img_green):
        # global self.isActualLight, self.lightcolor
        numRed = cv2.countNonZero(hsv_img_red)
        numGreen = cv2.countNonZero(hsv_img_green)

        if (numRed > lightthresh) or (numGreen > lightthresh):  # lightlimit >
            self.isActualLight = True
            if numGreen < numRed:
                self.lightcolor = False
            else:
                self.lightcolor = True

    def setLOI(self):
        # global self.LOIymin,self.LOIymax,self.LOIxmin,self.LOIxmax, self.LOIcolor, self.LOIcenterx, self.LOIcentery,self.lightcolor
        if not self.lightcolor:
            self.LOIcolor = False
        else:
            self.LOIcolor = True
        self.LOIymin = ymin
        self.LOIymax = ymax
        self.LOIxmin = xmin
        self.LOIxmax = xmax
        self.LOIcenterx = center
        self.LOIcentery = centery

    def getLOI(self, ymin):
        # global self.LOIymin,self.lightcolor,self.doublelight,self.prevdoublelight
        if not self.prevdoublelight:
            if self.doublelight and not self.lightcolor:
                self.prevdoublelight = True
            if ymin < self.LOIymin or self.prevdoublelight:
                self.setLOI()
        else:
            if self.doublelight and ymin < self.LOIymin:
                self.setLOI()

    def drawLights(self, mode):
        global frameout  # , self.isActualLight, self.doublelight, self.lightcolor, self.LOIs,debugDisplay
        if debugDisplay:
            # mode 0 for frame draw and
            # mode 1 for time draw
            if mode == 0:
                label = "%s: %d%%" % (
                    object_name,
                    int(scores[i] * 100),
                )  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )  # Get font size
                label_ymin = max(
                    ymin, labelSize[1] + 10
                )  # Make sure not to draw label too close to top of window
                if self.doublelight:
                    cv2.rectangle(frameout, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                    cv2.rectangle(
                        frameout,
                        (xmin, label_ymin - labelSize[1] - 10),
                        (xmin + labelSize[0], label_ymin + baseLine - 10),
                        (0, 255, 255),
                        cv2.FILLED,
                    )  # Draw white box to put label text in
                    cv2.putText(
                        frameout,
                        "%s%%:%d%%" % (self.lightxt, int(scores[i] * 100)),
                        (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                    )  # Draw label text
                elif self.isActualLight:

                    if self.lightcolor:
                        # Example: 'person: 72%'
                        label = "Green: %d%%" % (int(scores[i] * 100))
                        labelSize, baseLine = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )  # Get font size
                        cv2.rectangle(
                            frameout, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2
                        )
                        cv2.rectangle(
                            frameout,
                            (xmin, label_ymin - labelSize[1] - 10),
                            (xmin + labelSize[0], label_ymin + baseLine - 10),
                            (0, 255, 0),
                            cv2.FILLED,
                        )  # Draw white box to put label text in
                        cv2.putText(
                            frameout,
                            "Green:%d%%" % (int(scores[i] * 100)),
                            (xmin, label_ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 0),
                            2,
                        )  # Draw label text
                    else:
                        # Example: 'person: 72%'
                        label = "Red: %d%%" % (int(scores[i] * 100))
                        labelSize, baseLine = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )  # Get font size
                        cv2.rectangle(
                            frameout, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2
                        )
                        cv2.rectangle(
                            frameout,
                            (xmin, label_ymin - labelSize[1] - 10),
                            (xmin + labelSize[0], label_ymin + baseLine - 10),
                            (0, 0, 255),
                            cv2.FILLED,
                        )  # Draw white box to put label text in
                        cv2.putText(
                            frameout,
                            "Red:%d%%" % (int(scores[i] * 100)),
                            (xmin, label_ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 0),
                            2,
                        )  # Draw label text
            if mode == 1:
                # draw light of intrest as white
                if self.LOIxmax:
                    cv2.rectangle(
                        frameout,
                        (self.LOIxmin, self.LOIymin),
                        (self.LOIxmax, self.LOIymax),
                        (255, 255, 255),
                        2,
                    )

                if self.reachedr:
                    cv2.putText(
                        frameout,
                        "Red",
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )
                if self.reachedg:
                    cv2.putText(
                        frameout,
                        "Green",
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frameout,
                    "%s" % (self.oldLOIcolor),
                    (int(vwidth - 70), 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

    def threshColor(self):
        # global self.oldLOIcolor,self.reachedg,self.reachedr
        # desc:  Average many frames of light detection to remove random noise by using a single value (self.oldLOIcolor)
        #        with pos beign green, neg red and zero no light
        if self.isActualLight:  # if in the frame a light is found
            if self.LOIcolor:  # if the most important light is green
                # if previous x(thresh) frames had the same light
                if self.oldLOIcolor > lightChangeThresh:
                    self.reachedg = True
                    # print("1")
                else:  # else add it to the list
                    self.oldLOIcolor = self.oldLOIcolor + 1
                    # print("2")
            else:  # if that light is red
                # if previous x(thresh) frames had the same light
                if self.oldLOIcolor < -lightChangeThresh:
                    self.reachedr = True
                    # print("3")
                else:  # else add it to the list
                    self.oldLOIcolor = self.oldLOIcolor - 1
                    # print("4")
        else:  # if there is no light in the frame
            if self.oldLOIcolor > 0:  # if previous frames had green light
                self.oldLOIcolor = self.oldLOIcolor - 1  # remove a frame from history
                # print("5")
            if self.oldLOIcolor < 0:  # if previous frames had red light
                # print("6")
                self.oldLOIcolor = self.oldLOIcolor + 1  # remove a frame from history
        if self.oldLOIcolor == 0:
            self.reachedg = False
            self.reachedr = False
        if RaspberryPi:
            if self.reachedr:
                statled.light = 1
        return self.reachedg, self.reachedr

    def runRedWarn(self):
        global frameout, timeDistLight, carSpeed  # ,self.warntime,self.redWarn
        # desc: Check if red has reached threshhold and if the position of the light of intrest is closing on the edge and
        #        deceleration is not detected Warn the driver

        # check if is red light and the position of the light of intrest is closing on the edge
        if (
            self.reachedr
            and carSpeed > 5
            and (
                vwidth - egdeThresh <= self.LOIcenterx
                or self.LOIcenterx <= egdeThresh
                or egdeThresh >= self.LOIcentery
            )
        ):
            self.redWarn = True
            self.warnStart = (
                cv2.getTickCount()
            )  # store the last time a red light was detected
            timeDistLight = minDistLight / (
                carSpeed / 3.6
            )  # calculate the time before the vehicle has passed the light based on speed
            print("warn")
            # cv2.putText(frameout, 'Warn', (int(vwidth / 2 + 170), 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
            #             (0, 0, 255), 3, cv2.LINE_AA)
        else:
            self.redWarn = False

        warncheck = cv2.getTickCount()
        self.warntime = (
            warncheck - self.warnStart
        ) / freq  # find the time since a red light was detected
        # print(self.warntime, timeDistLight)
        # if Deceleration is bellow a threshold and (red light is detected or time since a red light was detected is
        # smaller than the time it takes to pass the light) then warn the driver
        if (self.redWarn or self.warntime <= timeDistLight) and carAccel > accelthresh:
            print("RUN RED ALERT")
            frameout = warn(4, frameout)


def warn(arg, mask):
    global warned
    if arg == 1 and (not tsign_left and not tsign_right) and carSpeed > 20:
        # print("LINE CROSSED")
        cv2.putText(
            mask,
            "LINE CROSSED",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        if not mixer.get_busy():
            sound.play()
        # beep.play(1)
    if arg == 2:
        cv2.putText(
            mask,
            "COLLISION ALERT",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        if not mixer.get_busy():
            sound.play()
        # beep.play(2)
        warned = True
    if arg == 3:
        cv2.putText(
            mask,
            "DEPARTURE ALERT",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        if not mixer.get_busy():
            sound.play()
        # beep.play(1)
        warned = True
    if arg == 4:
        cv2.putText(
            mask,
            "RUN RED ALERT",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        if not mixer.get_busy():
            sound.play()
        # beep.play(2)
        warned = True

    return mask


try:
    if not videoFileMode:
        # Ustawienie metody startu dla procesów
        multiprocessing.set_start_method('spawn', force=True)
        obdii = OBDII()
        print("Inicjalizacja systemu OBD...")
        obdii.connectOBD()

        # 2. Start animacji LED
        if RaspberryPi:
            loading_animation = AsyncLoadingAnimation(num_leds=8)
            loading_animation.start()

        # 3. Start kamery (teraz nie będzie konfliktu z Tkinterem)
        if videoFileMode:
            videostream = cv2.VideoCapture(nameOfFile)
        else:
            videostream = VideoStream().start()
        skip = 0
        frameskip = 1
    vwidth = int(videostream.get(cv2.CAP_PROP_FRAME_WIDTH))
    vheight = int(videostream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # initialize buzzer and sound
    if RaspberryPi:
        beep = Beep()
    mixer.init()
    sound = mixer.Sound("/home/edex/Desktop/Pi-ADAS/warnSound.ogg")

    # initialize Status LEDs
    if RaspberryPi:
        time.sleep(0.5)
        statled = statLED()

    # initialize Hall effect sensor for turn signal
    if RaspberryPi:
        turn = hallSens()

    # initialize model with arguments
    Model = Model()
    loading_animation.stop()

    # timestamp an output directory for each capture
    # outdir = args.output_path

    # Initialize frame rate calculation
    fi = []
    fc = []
    ft = []
    frame_rate_infr = 1
    frame_rate_calc = 1
    frame_rate_tot = 1
    freq = cv2.getTickFrequency()

    # Initrialize Lane Departure
    laneDep = LaneDeparture(videostream)

    # Initrialize Object colision
    objCol = objColision()

    # Initrialize Light Detection
    lightDet = LightDet()

    tcalc = 0

    frameCount = 0
    oldframe = []
    last_settings_state = False

    def on_touch(event, x, y, flags, param):
        global show_settings
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = button_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                show_settings = not show_settings
    
    
    while True:
        if show_settings != last_settings_state:
            if show_settings:
                init_trackbars()
                print("Otwarto okno ustawień")
            else:
                try:
                    cv2.destroyWindow("settings")
                    print("Zamknięto okno ustawień")
                except:
                    pass
            last_settings_state = show_settings
        # Grab frame from video stream
        # print("\n---Getting next frame "+str(frameCount)+"---")
        ret, frame = videostream.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            print("Reached the end of the video!")
            break
        # cv2.imshow('image', frame)
        frameCount = frameCount + 1
        # if videoFileMode:
        #     carSpeed = avglog[frameCount]
        if skip <= 0 and frameCount % frameskip == 0:
            if show_settings:
                try:
                    # Sprawdzamy czy okno nadal żyje zanim pobierzemy dane
                    if cv2.getWindowProperty("settings", cv2.WND_PROP_VISIBLE) >= 1:
                        CollisionSens = cv2.getTrackbarPos("CollisionSens", "settings") * 0.01
                        CollisionThresh = cv2.getTrackbarPos("CollisionThresh", "settings") * 1000
                        departureSens = cv2.getTrackbarPos("departureSens", "settings")
                        minDistLight = cv2.getTrackbarPos("minDistLight", "settings")
                        accelN = cv2.getTrackbarPos("accelN", "settings")
                except:
                    # Jeśli okno zostało zamknięte "krzyżykiem", zresetuj flagę
                    show_settings = False

            if tcalc == 0:
                tstart = cv2.getTickCount()
            else:
                tstart = tcalc

            tinfr = cv2.getTickCount()

            frameor = frame

            vwidth = int(videostream.get(cv2.CAP_PROP_FRAME_WIDTH))
            vheight = int(videostream.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # if videoFileMode:
            if vheight != vwidth:
                if vheight > vwidth:
                    cropstart = (vheight / 2 - vwidth / 2) - cropoffset
                    cropend = (vheight / 2 + vwidth / 2) - cropoffset
                    frame = frame[int(cropstart) : int(cropend), 0 : int(vwidth)]
                    vheight = vwidth
                else:
                    cropstart = vwidth / 2 - vheight / 2
                    cropend = vwidth / 2 + vheight / 2
                    frame = frame[0 : int(vheight), int(cropstart) : int(cropend)]
                    vwidth = vheight

            frameout = frame

            print("\n---Prossesing next frame---")
            tstart = cv2.getTickCount()
            warned = False
            if not videoFileMode:
                obdii.UpdateAccelSpeed()

            if RaspberryPi:
                turn.setTurn()

            # Perform Lane departure warning

            # try:
            Thread(target=laneDep.laneDeparture, args=(frameor,)).start()
            # except:

            # laneframe = laneDep.laneDeparture(frameor)

            # Perform object detection by running the model with the image as input
            boxes, classes, scores = Model.detectObj(frame)

            tinfr = cv2.getTickCount()

            print("-Done infrancing")

            # initialize Frame Data
            objCol.initFrameVar()

            lightCenters = []
            lightDet.initFrameVar()

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                # num=len(scores)
                if (scores[i] > Model.min_conf_threshold) and (scores[i] <= 1.0):
                    # Look up object name from "labels" array using class index
                    object_name = Model.labels[int(classes[i])]

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * Model.imH)))
                    xmin = int(max(1, (boxes[i][1] * Model.imW)))
                    ymax = int(min(Model.imH, (boxes[i][2] * Model.imH)))
                    xmax = int(min(Model.imW, (boxes[i][3] * Model.imW)))

                    # find area,center,lenghts
                    xlenght = xmax - xmin
                    ylenght = ymax - ymin
                    area = (xlenght) * (ylenght)
                    center = xmin + (xlenght) / 2
                    centery = ymin + (ylenght) / 2

                    # ------------ Frame - Traffic light color -------------------

                    if object_name == "traffic light":
                        # ignore lights in the same possition
                        if all(
                            abs(center - lightCenter) > xlenght
                            for lightCenter in lightCenters
                        ):
                            lightCenters.append(center)

                            # Get Color Windows of traffic Light
                            hsv_img_red, hsv_img_green = lightDet.getLightColorWindows(
                                frame
                            )

                            # check for double light
                            lightDet.checkDoubleLight(hsv_img_red, hsv_img_green)

                            # check for normal lights
                            if not lightDet.doublelight:
                                lightDet.checkNormalight(hsv_img_red, hsv_img_green)

                            # Get light color of intrest
                            lightDet.getLOI(ymin)

                            # Draw label
                            lightDet.drawLights(0)

                    # ------------ Frame - Colision Detection-------------------

                    else:
                        # desc pt1-Frame:Select a car as car of intrest by which are in path and closest, also check if its a new car

                        # ignore boxes that are touching the edge of the frame (cant judge size)
                        if (xmin > 10) and (xmax < vwidth - 10):

                            # Select a car as car of intrest by which are in path and closest, also check if its a new car
                            objCol.getObjOfIntrest()

                            # Draw labels
                            objCol.drawObjCol(0)

            #  Time - Light of intrest actions

            # lightDet.lightTracking()

            # draw light of intrest as white
            lightDet.drawLights(1)

            # desc:  Average many frames of light detection to remove random noise by using a single value (oldLOIcolor)
            #        with positive being green, negative red and zero no light
            lightDet.threshColor()

            # desc: Check if red has reached threshhold and based on the position of the light of intrest and the
            #        deceleration warn the driver
            lightDet.runRedWarn()

            # Time - Colision Detection

            # desc pt2-Time: Find if there is a collision or departure if stoped by mesuring the change in size of the car in front
            #       (coi) and comparing it to a threshhold but also taking into account if threre is a new car
            objCol.objColWarn()

            objCol.drawObjCol(1)

            print("-Done calculating")

            if debugDisplay:
                # 1. Przygotowanie obrazów (skalowanie do wspólnej wysokości 480px)
                h_target = 480
                
                # Pobieramy obrazy z klas
                img_left = laneDep.laneframe if len(laneDep.laneframe) > 0 else np.zeros((h_target, 400, 3), np.uint8)
                img_right = frameout
                
                # Skalowanie
                img_left_res = cv2.resize(img_left, (int(img_left.shape[1] * (h_target/img_left.shape[0])), h_target))
                img_right_res = cv2.resize(img_right, (int(img_right.shape[1] * (h_target/img_right.shape[0])), h_target))
                
                # Łączenie w jeden poziom (Landscape | ADAS)
                combined_view = np.hstack((img_left_res, img_right_res))

                # 2. Rysowanie przycisku SETTINGS
                bx, by, bw, bh = button_rect
                color = (0, 255, 0) if show_settings else (100, 100, 100)
                cv2.rectangle(combined_view, (bx, by), (bx + bw, by + bh), color, -1)
                cv2.putText(combined_view, "SETTINGS", (bx + 15, by + 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 3. Wyświetlanie głównego okna i podpięcie myszy/dotyku
                cv2.namedWindow("ADAS_SYSTEM", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("ADAS_SYSTEM", on_touch)
                cv2.imshow("ADAS_SYSTEM", combined_view)
                
                # Zamknij stare pojedyncze okna, jeśli istnieją (sprzątanie)
                if cv2.getWindowProperty("ADAS", 0) >= 0: cv2.destroyWindow("ADAS")
                if cv2.getWindowProperty("lanedep", 0) >= 0: cv2.destroyWindow("lanedep")
                if cv2.getWindowProperty("frameout", 0) >= 0: cv2.destroyWindow("frameout")

            if RaspberryPi:
                if tsign_left:
                    statled.set_state(turnl=1)
                    statled.update_leds()
                if tsign_right:
                    statled.set_state(turnr=1)
                    statled.update_leds()

                if (frameCount / frameskip) % 3 == 0:
                    statled.update_leds()

            # save frames
            if saveFramestoDisk:
                path = str(outdir) + "/" + str(datetime.datetime.now()) + ".jpg"
                Thread(target=cv2.imwrite, args=(path, frame)).start()
                # print(path, status)status =

            # Calculate framerates

            tcalc = cv2.getTickCount()
            timeinfr = (tinfr - tstart) / freq
            timecalc = (tcalc - tinfr) / freq
            timetotal = (tcalc - tstart) / freq
            frame_rate_infr = 1 / timeinfr
            frame_rate_calc = 1 / timecalc
            frame_rate_tot = 1 / timetotal
            fi.append(frame_rate_infr)
            fc.append(frame_rate_calc)
            ft.append(frame_rate_tot)
            print(
                "Infrancing: {0:.2f} fps".format(frame_rate_infr)
                + " ({0:.2f}s)".format(timeinfr)
            )
            print(
                "Calculation: {0:.2f} fps".format(frame_rate_calc)
                + " ({0:.2f}s)".format(timecalc)
            )
            print(
                "Total: {0:.2f} fps".format(frame_rate_tot)
                + " ({0:.2f}s)".format(timetotal)
            )

            if videoFileMode and (frameByframe or warned) and oldframe != []:
                cv2.imshow("prev frame", oldframe)
                # cv2.waitKey(0) == ord('e')

            oldframe = frameout

            if cv2.waitKey(1) == ord("q"):
                break
            if cv2.waitKey(1) == ord("a"):
                tsign_left = True
                tsign_right = False
            if cv2.waitKey(1) == ord("d"):
                tsign_right = True
                tsign_left = False
            if cv2.waitKey(1) == ord("w"):
                tsign_right = False
                tsign_left = False
            # Press 'q' to quit

        else:
            skip = skip - 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


finally:
    if RaspberryPi and "statled" in locals():
        statled.cleanup()
    # Clean up
    cv2.destroyAllWindows()
    if not videoFileMode:
        # with open('speed1.log','wb') as fp:
        #     pickle.dump(obdii.speedlog,fp)
        # with open('avgaccl1.log','wb') as fp:
        #     pickle.dump(obdii.avglog,fp)
        if "videostream" in locals() and videostream is not None:
            videostream.stop()
    print("\n----Exiting Program----")
    if len(fi) > 0:
        print(
            "Average Infrancing: {0:.2f}".format(sum(fi) / len(fi))
            + "fps"
            + " ({0:.2f}s)".format(1 / (sum(fi) / len(fi)))
        )
        print(
            "Average Calculation: {0:.2f} fps".format(sum(fc) / len(fc))
            + " ({0:.2f}s)".format(1 / (sum(fc) / len(fc)))
        )
        print(
            "Total Average: {0:.2f} fps".format(sum(ft) / len(ft))
            + " ({0:.2f}s)\n\n\n".format(1 / (sum(ft) / len(ft)))
        )
