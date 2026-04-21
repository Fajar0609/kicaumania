import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import time
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Kicau Mania Detector", page_icon="🐾")
st.title("🐾 Kicau Mania Web Detector")
st.markdown("""
Aplikasi ini mendeteksi:
1. **Dua tangan** menutupi area mulut.
2. **Kibasan tangan** (gerakan kanan/kiri) saat mulut tertutup.
3. **Trigger** akan memutar video otomatis.
""")

# --- KONSTANTA LOGIKA ---
CAT_VIDEO_PATH = "kicau-mania.mp4" 
WAVE_THRESHOLD = 1
WAVE_AMPLITUDE = 0.01
MOUTH_COVER_DISTANCE = 0.35
COVER_WAVE_WINDOW = 5.0
PLAY_TIMEOUT = 1.0

# --- SETUP MEDIAPIPE (Global agar tidak berat) ---
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Inisialisasi model
hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
face_mesh_detector = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# --- CLASS DETEKTOR ---
class WaveDetector:
    def __init__(self):
        self.last_x = None
        self.peak_x = None
        self.direction = 0  
        self.direction_count = 0
        self.last_move_time = time.time()

    def reset(self):
        self.last_x = None
        self.peak_x = None
        self.direction = 0
        self.direction_count = 0
        self.last_move_time = time.time()

    def is_moving(self):
        return (time.time() - self.last_move_time) < PLAY_TIMEOUT

    def update(self, x_position):
        now = time.time()
        if self.last_x is None:
            self.last_x = x_position
            self.peak_x = x_position
            return False
        delta = x_position - self.last_x
        self.last_x = x_position
        if abs(delta) > 0.005:
            self.last_move_time = now
            new_direction = 1 if delta > 0 else -1
            if self.direction != new_direction and self.direction != 0:
                if abs(x_position - self.peak_x) >= WAVE_AMPLITUDE:
                    self.direction_count += 1
                self.peak_x = x_position
            self.direction = new_direction
        is_wave = self.direction_count >= WAVE_THRESHOLD
        if is_wave: self.reset()
        return is_wave

# --- WORKER PEMROSES VIDEO ---
class KicauProcessor(VideoTransformerBase):
    def __init__(self):
        self.wave_left = WaveDetector()
        self.wave_right = WaveDetector()
        self.last_mouth_cover_time = 0
        self.is_playing = False
        self.last_movement_time = 0
        
        # Load video trigger
        self.video_cap = cv2.VideoCapture(CAT_VIDEO_PATH)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        hands_result = hands_detector.process(rgb_frame)
        face_result = face_mesh_detector.process(rgb_frame)

        hand_centers = []
        mouth_center = None
        now = time.time()

        # 1. Deteksi Mulut
        if face_result.multi_face_landmarks:
            face_landmarks = face_result.multi_face_landmarks[0]
            upper = face_landmarks.landmark[13]
            lower = face_landmarks.landmark[14]
            mouth_center = ((upper.x + lower.x)/2, (upper.y + lower.y)/2)
            cv2.circle(img, (int(mouth_center[0]*w), int(mouth_center[1]*h)), 10, (255, 255, 255), 2)

        # 2. Deteksi Tangan
        if hands_result.multi_hand_landmarks:
            for res in hands_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                wrist = res.landmark[0]
                hand_centers.append((wrist.x, wrist.y))

        # 3. Logika Trigger
        hand_centers.sort(key=lambda p: p[0])
        
        # Cek Tutup Mulut
        mouth_covered = False
        if mouth_center and len(hand_centers) >= 2:
            for hx, hy in hand_centers:
                if np.hypot(hx - mouth_center[0], hy - mouth_center[1]) <= MOUTH_COVER_DISTANCE:
                    mouth_covered = True
                    break
        
        if mouth_covered:
            self.last_mouth_cover_time = now
        
        mouth_recent = (now - self.last_mouth_cover_time) <= COVER_WAVE_WINDOW

        # Cek Kibasan
        wave_now = False
        if len(hand_centers) > 0:
            w_l = self.wave_left.update(hand_centers[0][0])
            w_r = self.wave_right.update(hand_centers[1][0]) if len(hand_centers) > 1 else False
            wave_now = w_l or w_r

        # Status Trigger
        if len(hand_centers) >= 2 and mouth_recent and wave_now and not self.is_playing:
            self.is_playing = True
            self.last_movement_time = now
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 4. Overlay Video Jika Aktif
        if self.is_playing:
            if self.wave_left.is_moving() or self.wave_right.is_moving():
                self.last_movement_time = now
            
            if now - self.last_movement_time > PLAY_TIMEOUT:
                self.is_playing = False
            else:
                ret_v, v_frame = self.video_cap.read()
                if not ret_v:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_v, v_frame = self.video_cap.read()
                
                if ret_v:
                    # Resize video agar pas di pojok atau menutupi layar
                    v_frame = cv2.resize(v_frame, (w // 2, h // 2))
                    img[0:h//2, 0:w//2] = v_frame
                    cv2.putText(img, "KICAU MODE ON", (10, h//2 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

# --- MENJALANKAN WEBRTC ---
webrtc_streamer(
    key="kicaumania",
    video_transformer_factory=KicauProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)