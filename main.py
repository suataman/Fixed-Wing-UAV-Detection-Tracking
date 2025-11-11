import math
import cv2
import time
from ultralytics import YOLO

# Model ve video path
model_path = r"models/model.pt"


cap = cv2.VideoCapture(r"videolar\example_video.mp4")
#cap = cv2.VideoCapture(0) # if you want to use webcam

# for YOLOV8 Model
model = YOLO(model_path)

# OpenCV CSRT tracker 
tracker = cv2.legacy.TrackerCSRT_create()


init_tracking = False
bbox = None
tracked_conf = None
#FPS
fps_list = []
video_fps = cap.get(cv2.CAP_PROP_FPS)
prev_frame_time = time.time()
new_frame_time = 0
#YAW PİTCH
vector_length = 100
yaw_factor = 0.05
pitch_factor = 0.05
#locking
tracking_start_time = None
timeout = 5  # seconds
min_distance = None

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1080, 720))
    visual = img.copy()
    height, width, _ = img.shape


    rect_width = int(width * 0.50)  # Vertical %50
    rect_height = int(height * 0.80)  # Horizontal %80
    top_left = (int((width - rect_width) / 2), int((height - rect_height) / 2))  
    bottom_right = (top_left[0] + rect_width, top_left[1] + rect_height)
    cv2.rectangle(visual, top_left, bottom_right, (0, 255, 255), 2)

    center_x = width // 2
    center_y = height // 2
    cv2.circle(visual, (center_x, center_y), 5, (0, 0, 0), cv2.FILLED)

    if not init_tracking or time.time() - tracking_start_time >= timeout:
        closest_box = None
        min_distance = float("inf")
        closest_conf = 0

        results = model.predict(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                if conf < 0.4:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                distance = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_box = (x1, y1, w, h)
                    closest_conf = conf

        if closest_box is not None:
            bbox = closest_box
            tracked_conf = closest_conf
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(img, bbox)
            init_tracking = True
            tracking_start_time = time.time()
            cv2.putText(visual, f"Distance: {min_distance:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    else:
        success, bbox = tracker.update(img)
        remaining_time = max(0, timeout - (time.time() - tracking_start_time))
        cv2.putText(visual, f"Tracker reset in: {remaining_time:.1f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            obj_center_x = x + w // 2
            obj_center_y = y + h // 2

            yaw_angle = (obj_center_x - center_x) * yaw_factor
            pitch_angle = (obj_center_y - center_y) * pitch_factor
            end_x = int(center_x + yaw_angle * vector_length)
            end_y = int(center_y + pitch_angle * vector_length)
            cv2.line(visual, (center_x, center_y), (end_x, end_y), (255, 0, 255), 7)
            cv2.circle(visual, (end_x, end_y), 5, (255, 0, 255), -1)

            # Yaw and pitch value display
            cv2.putText(visual, f"Yaw: {yaw_angle:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(visual, f"Pitch: {pitch_angle:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            cv2.rectangle(visual, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(visual, f"Tracking Fixed-Wing UAV ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(visual, f"Detection Conf: {tracked_conf}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),2)
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(visual, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
            cv2.line(visual, (cx, cy), (center_x, center_y), (0, 0, 0), 2)
            cv2.putText(visual, f"Distance: {min_distance:.2f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),2)
        else:
            init_tracking = False

    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time

    fps_list.append(fps)
    if len(fps_list) > 10:  
        fps_list.pop(0)
    smoothed_fps = sum(fps_list) / len(fps_list)  # avarage FPS



    cv2.putText(visual, f"FPS: {int(smoothed_fps)} / Video FPS: {int(video_fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Predicted and tracking", visual)
    cv2.imshow("clean img", img)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()