import cv2
import time
from ultralytics import YOLO
import requests
from picamera2 import Picamera2

# ---------------- CONFIGURATION ----------------
# MODEL_PATH = "bulky_waste_dumping_yolov11/yolo11n.pt"
MODEL_PATH = "bulky_waste_dumping_yolov11/best.pt"
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_TOWN_COUNCIL_GROUP_ID"
TARGET_CLASSES = ["bulky_waste", "bin_full", "normal_waste"]
# -----------------------------------------------


picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

model = YOLO(MODEL_PATH)

detection_start_time = None
alert_sent = False


def send_telegram_alert(image_path, message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, "rb") as photo:
        requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": message},
            files={"photo": photo},
        )


print("Camera started. Running detection loop...")

while True:
    try:
        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run inference
        results = model(frame_bgr, imgsz=320, verbose=False)[0]
        annotated_frame = results.plot()

        detected = False
        target_name = ""  # Store the detected target class name

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Trigger only when the detected class is inside TARGET_CLASSES
            if class_name in TARGET_CLASSES:
                detected = True
                target_name = class_name
                # Exit loop once a valid target is detected
                break

        # --- UI State Machine & Timer Logic ---
        ui_status = "NORMAL"
        ui_color = (0, 255, 0)  # Green (BGR format)
        ui_subtext = "System Active & Scanning"

        if detected:
            if detection_start_time is None:
                detection_start_time = time.time()
                print(f"Detected: {target_name}. Starting 30s timer...")

            elapsed_time = time.time() - detection_start_time
            remaining_time = max(0, 30 - int(elapsed_time))

            if not alert_sent:
                if remaining_time > 0:
                    ui_status = "WARNING"
                    ui_color = (0, 165, 255)  # Orange
                    ui_subtext = (
                        f"Target Locked ({target_name}). "
                        f"Alert in: {remaining_time}s"
                    )
                else:
                    # Timer completed, trigger alert
                    img_path = "evidence.jpg"
                    results.save(filename=img_path)

                    send_telegram_alert(
                        img_path,
                        f"⚠️ Alert: {target_name} detected continuously for 30s."
                    )

                    alert_sent = True
                    print("Telegram alert sent!")

                    ui_status = "ALERT"
                    ui_color = (0, 0, 255)  # Red
                    ui_subtext = "EVIDENCE DISPATCHED TO TC"
            else:
                # Alert already sent, keep ALERT status active
                ui_status = "ALERT"
                ui_color = (0, 0, 255)
                ui_subtext = "EVIDENCE DISPATCHED TO TC"

        else:
            # Reset timer if the target disappears from the frame
            detection_start_time = None
            alert_sent = False

        # --- Draw UI Overlay (HUD) ---

        # 1. Draw semi-transparent black background panel
        #    to improve text visibility
        overlay = annotated_frame.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (450, 90),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.6,
            annotated_frame,
            0.4,
            0,
            annotated_frame
        )

        # 2. Draw main status title
        cv2.putText(
            annotated_frame,
            f"STATUS: {ui_status}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            ui_color,
            3
        )

        # 3. Draw subtitle (countdown or info text)
        cv2.putText(
            annotated_frame,
            ui_subtext,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # ---------------------------

        # Display final annotated frame
        cv2.imshow("Smart Estate Monitor - Edge UI", annotated_frame)

        # Use waitKey instead of time.sleep
        # Refresh every 500ms (2 FPS) to balance smoothness
        # and prevent Raspberry Pi 5 overheating
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)