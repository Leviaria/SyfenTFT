from adb_shell.adb_device import AdbDeviceTcp
from loguru import logger
import os
import cv2
import easyocr
import torch

SCREENSHOT_DIR = "screenshots"
SCREENSHOT_FILE = os.path.join(SCREENSHOT_DIR, "gold_screenshot.png")

FIXED_COORDS = {"x1": 1190, "y1": 623, "x2": 1238, "y2": 655}

def ensure_screenshot_dir():
    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)
        logger.info(f"Created directory: {SCREENSHOT_DIR}")

def connect_to_adb():
    device = AdbDeviceTcp("127.0.0.1", 5555)
    device.connect()
    logger.info("Connected to ADB.")
    return device

def take_screenshot(device, output_file=SCREENSHOT_FILE):
    temp_remote_path = "/sdcard/screenshot.png"
    device.shell(f"screencap -p {temp_remote_path}")
    device.pull(temp_remote_path, output_file)
    logger.info(f"Screenshot saved to: {output_file}")

def crop_image_cv(image_path, coords):
    image = cv2.imread(image_path)
    cropped = image[coords["y1"]:coords["y2"], coords["x1"]:coords["x2"]]
    cropped_file = os.path.join(SCREENSHOT_DIR, "cropped_region.png")
    cv2.imwrite(cropped_file, cropped)
    logger.info(f"Cropped region saved to: {cropped_file}")
    return cropped_file

def read_number(image_path, reader):
    results = reader.readtext(image_path, detail=0)
    for result in results:
        if result.strip().isdigit():
            return int(result.strip())
    raise ValueError("No numeric value found in the region.")

def main():
    try:
        ensure_screenshot_dir()
        device = connect_to_adb()
        take_screenshot(device)

        logger.info(f"Using fixed coordinates: {FIXED_COORDS}")
        cropped_file = crop_image_cv(SCREENSHOT_FILE, FIXED_COORDS)

        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        gold_amount = read_number(cropped_file, reader)

        logger.success(f"Gold amount detected: {gold_amount}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
