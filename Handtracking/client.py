import cv2
import numpy as np
import mediapipe as mp
import asyncio
import websockets
import threading
import queue
import time

SERVER_URL = "ws://localhost:8000/api/ws/video"

# ================= Mediapipe setup =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Queue chỉ giữ 1 frame mới nhất (no backlog)
feature_queue = queue.Queue(maxsize=1)

def extract_features(frame):
    """Trích xuất vector 42D từ frame"""
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        coords = []
        for p in lm.landmark:
            coords.extend([p.x, p.y])
        return np.array(coords, dtype=np.float32)
    else:
        return np.zeros(42, dtype=np.float32)  # padding nếu ko có tay

# ================= Thread A: Capture & Feature Extraction =================
def camera_thread():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        feat = extract_features(frame)

        # push feature vào queue (luôn giữ mới nhất)
        if not feature_queue.empty():
            try:
                feature_queue.get_nowait()  # bỏ cái cũ
            except queue.Empty:
                pass
        feature_queue.put_nowait(feat.tolist())

        cv2.imshow("Client", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================= Thread B: Async WebSocket =================
async def send_features():
    async with websockets.connect(SERVER_URL) as ws:
        while True:
            try:
                feat = feature_queue.get_nowait()  # lấy frame mới nhất
            except queue.Empty:
                await asyncio.sleep(0.005)  # không block, CPU nhẹ
                continue

            arr = np.array(feat, dtype=np.float32)
            await ws.send(arr.tobytes())

            # nhận response nếu có
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                print("Server:", response)
            except asyncio.TimeoutError:
                pass

            # hạn chế spam (ví dụ ~30fps = 0.033s)
            await asyncio.sleep(0.033)

# ================= Main =================
if __name__ == "__main__":
    # chạy camera thread song song
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    # chạy websocket loop
    asyncio.run(send_features())

# CONCLUSION : CODE IS OK
