from adb_shell.adb_device import AdbDeviceTcp
from loguru import logger
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import threading
import time
from functools import lru_cache

SCREENSHOT_DIR = "screenshots"
SCREENSHOT_FILE = os.path.join(SCREENSHOT_DIR, "gold_screenshot.png")
FIXED_COORDS = {"x1": 1190, "y1": 615, "x2": 1240, "y2": 665}
device_lock = threading.Lock()
device = None

def ensure_screenshot_dir():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

@lru_cache(maxsize=1)
def get_device():
    global device
    if device is None:
        with device_lock:
            if device is None:
                device = AdbDeviceTcp("127.0.0.1", 5555)
                device.connect()
    return device

def take_screenshot(output_file=SCREENSHOT_FILE):
    device = get_device()
    temp_remote_path = "/sdcard/screenshot.png"
    with device_lock:
        result = device.shell("screencap -p")
        if isinstance(result, bytes):
            with open(output_file, 'wb') as f:
                f.write(result)
        else:
            device.shell(f"screencap -p {temp_remote_path}")
            device.pull(temp_remote_path, output_file)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contrast = cv2.convertScaleAbs(thresh, alpha=1.5, beta=0)
    return cv2.fastNlMeansDenoising(contrast)

def crop_and_process_image(image_path, coords):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")
    cropped = image[coords["y1"]:coords["y2"], coords["x1"]:coords["x2"]]
    processed = preprocess_image(cropped)
    cropped_file = os.path.join(SCREENSHOT_DIR, "cropped_region.png")
    cv2.imwrite(cropped_file, processed)
    return cropped_file, processed

def read_number(processed_image):
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    pil_image = Image.fromarray(processed_image)
    result = pytesseract.image_to_string(pil_image, config=custom_config).strip()
    for word in result.split():
        if word.isdigit():
            return int(word)
    raise ValueError("No numeric value found in the region.")

def main():
    try:
        ensure_screenshot_dir()
        take_screenshot()
        cropped_file, processed_image = crop_and_process_image(SCREENSHOT_FILE, FIXED_COORDS)
        start_time = time.time()
        gold_amount = read_number(processed_image)
        end_time = time.time()
        processing_time = end_time - start_time
        logger.success(f"Gold amount detected: {gold_amount}")
        logger.debug(f"Processing time: {processing_time:.2f} seconds")
        return gold_amount, processing_time
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None, None

if __name__ == "__main__":
    main()
