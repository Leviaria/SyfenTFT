from adb_shell.adb_device import AdbDeviceTcp
from loguru import logger
import os
import json
from datetime import datetime
import cv2
from PyQt6.QtWidgets import QApplication, QInputDialog

COORDS_FILE = "coords.json"

def connect_to_adb():
    device = AdbDeviceTcp("127.0.0.1", 5555)
    device.connect()
    logger.info("Connected to ADB.")
    return device

def take_screenshot(device, output_dir="screenshots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = os.path.join(output_dir, f"screenshot_{timestamp}.png")
    temp_remote_path = "/sdcard/screenshot.png"
    device.shell(f"screencap -p {temp_remote_path}")
    device.pull(temp_remote_path, local_path)
    logger.info(f"Screenshot saved to: {local_path}")
    return local_path

def get_region_name():
    app = QApplication([])
    name, ok = QInputDialog.getText(None, "Region Name", "Enter the name for this region:")
    if ok and name.strip():
        return name.strip()
    else:
        return "Unnamed"

def select_region(image_path):
    image = cv2.imread(image_path)
    clone = image.copy()
    coords = []
    dragging = False
    region_name = None

    def click_and_crop(event, x, y, flags, param):
        nonlocal coords, dragging, clone, region_name
        if event == cv2.EVENT_LBUTTONDOWN:
            coords = [(x, y)]
            dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            temp_clone = clone.copy()
            cv2.rectangle(temp_clone, coords[0], (x, y), (0, 255, 0), 2)
            width = abs(x - coords[0][0])
            height = abs(y - coords[0][1])
            size_text = f"{width}x{height}"
            cv2.putText(temp_clone, size_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("EasyOCR Screencoords", temp_clone)
        elif event == cv2.EVENT_LBUTTONUP:
            coords.append((x, y))
            cv2.rectangle(clone, coords[0], coords[1], (0, 255, 0), 2)
            dragging = False
            region_name = get_region_name()
            if region_name:
                center_x = (coords[0][0] + coords[1][0]) // 2
                center_y = min(coords[0][1], coords[1][1]) - 10
                cv2.putText(clone, region_name, (center_x - 50, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("EasyOCR Screencoords", clone)
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                region = {
                    "name": region_name,
                    "x1": min(x1, x2),
                    "y1": min(y1, y2),
                    "x2": max(x1, x2),
                    "y2": max(y1, y2)
                }
                save_to_json(region)
                logger.success(f"Region '{region['name']}' saved with coordinates: ({region['x1']}, {region['y1']}, {region['x2']}, {region['y2']})")
            cv2.imshow("EasyOCR Screencoords", clone)

    cv2.namedWindow("EasyOCR Screencoords")
    cv2.setMouseCallback("EasyOCR Screencoords", click_and_crop)

    while True:
        if cv2.getWindowProperty("EasyOCR Screencoords", cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.imshow("EasyOCR Screencoords", clone)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

def save_to_json(region, file=COORDS_FILE):
    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(region)

    with open(file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved region to {file}: {region}")

def print_regions(file=COORDS_FILE):
    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)
            logger.info("Regions:")
            for idx, reg in enumerate(data, start=1):
                logger.info(f"{idx}. Name: {reg['name']}, Coordinates: ({reg['x1']}, {reg['y1']}, {reg['x2']}, {reg['y2']})")

def main():
    try:
        device = connect_to_adb()
        screenshot_path = take_screenshot(device)
        logger.info("Select regions on the screenshot (Close the window to finish).")
        select_region(screenshot_path)
        print_regions()
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
