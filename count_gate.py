import cv2
import torch
import numpy as np
from collections import deque
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# ========== SETTINGS ==========
weights_path = "C:/Users/hatem/Desktop/project2/YOLO-CROWD/runs/train/continue_640_200/weights/best.pt"
video_path = "C:/Users/hatem/Downloads/test2.mp4"

conf_thres = 0.15
iou_thres = 0.35
imgsz = 640
stride = 32
CROSS_BUFFER = 25    # pixels below the line before counting
# ================================

def letterbox(img, new_shape=imgsz, stride=32, color=(114,114,114)):
    shape = img.shape[:2]
    r = min(new_shape/shape[0], new_shape/shape[1])
    new_unpad = int(round(shape[1]*r)), int(round(shape[0]*r))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw//2, dh//2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img

device = select_device('')
model = attempt_load(weights_path, map_location=device)

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

entry_line_y = height // 2
line_selected = False

def select_line(event, x, y, flags, param):
    global entry_line_y, line_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        entry_line_y = y
        line_selected = True
        print(f"✅ Crossing line set at Y = {entry_line_y}")

cv2.namedWindow('Set Line')
cv2.setMouseCallback('Set Line', select_line)
ret, frame = cap.read()
if not ret:
    print("Cannot open video")
    exit()
cv2.imshow('Set Line', frame)
print("🔴 Click on the frame where people cross, then press any key...")
cv2.waitKey(0)
cv2.destroyWindow('Set Line')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Improved tracking: store last center (x,y) and crossed flag
tracked = {}          # id -> {'last_center': (cx, cy), 'crossed': bool}
next_id = 0
total_count = 0
recently_crossed = deque(maxlen=40)

def is_near_recent_cross(cx, cy, thresh=50):
    for (x, y) in recently_crossed:
        if abs(cx - x) < thresh and abs(cy - y) < thresh:
            return True
    return False

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    img = letterbox(frame, new_shape=imgsz, stride=stride)
    img = img[:, :, ::-1].transpose(2,0,1)
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[0])

    detections = []  # (cx, cy, bbox)
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, (x1, y1, x2, y2)))

    if frame_idx % 30 == 0:
        print(f"Frame {frame_idx}: detected {len(detections)} persons")

    used_ids = set()
    for cx, cy, bbox in detections:
        best_id = None
        best_dist = 80
        for obj_id, data in tracked.items():
            if data['crossed']:
                continue
            last_cx, last_cy = data['last_center']
            dist = np.hypot(cx - last_cx, cy - last_cy)  # Euclidean distance
            if dist < best_dist:
                best_dist = dist
                best_id = obj_id

        if best_id is not None:
            # Existing object
            last_cx, last_cy = tracked[best_id]['last_center']
            # Count only when crossing from top to bottom (last_y < line <= current_y)
            if not tracked[best_id]['crossed'] and last_cy < entry_line_y <= cy:
                if not is_near_recent_cross(cx, cy, 60):
                    total_count += 1
                    tracked[best_id]['crossed'] = True
                    recently_crossed.append((cx, cy))
                    print(f"🚶 Person {best_id} crossed line (Y: {last_cy}→{cy}) Total: {total_count}")
                else:
                    print(f"⚠️ Duplicate ignored for {best_id}")
            # Update position
            tracked[best_id]['last_center'] = (cx, cy)
            used_ids.add(best_id)
            # Draw
            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f'ID:{best_id}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        else:
            # New object
            tracked[next_id] = {'last_center': (cx, cy), 'crossed': False}
            used_ids.add(next_id)
            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f'ID:{next_id}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            next_id += 1

    # Draw line and counter
    cv2.line(frame, (0, entry_line_y), (width, entry_line_y), (0,0,255), 3)
    cv2.putText(frame, f'CROSSED: {total_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f'DET: {len(detections)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow('Gate Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ FINAL COUNT: {total_count} people crossed the line.")