import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(page_title="Kicau Mania Detector", page_icon="🐾")
st.title("🐾 Kicau Mania Detector")
st.text("Pastikan webcam aktif. Gunakan dua tangan untuk trigger video!")

# --- KONSTANTA ---
CAT_VIDEO_PATH = "kicau-mania.mp4" 
WAVE_THRESHOLD = 1
WAVE_AMPLITUDE = 0.01
MOUTH_COVER_DISTANCE = 0.35
COVER_WAVE_WINDOW = 5.0
PLAY_TIMEOUT = 0.8 # Disesuaikan sedikit untuk sinkronisasi web

# --- SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
face_mesh_detector = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# --- CLASS & FUNGSI HELPER ---
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

def get_hand_center(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    return wrist.x, wrist.y

def get_mouth_center(face_landmarks):
    upper = face_landmarks.landmark[13]
    lower = face_landmarks.landmark[14]
    return (upper.x + lower.x)/2, (upper.y + lower.y)/2

def is_mouth_covered(hand_centers, mouth_center):
    if not mouth_center or len(hand_centers) < 2: return False
    for hx, hy in hand_centers:
        if np.hypot(hx - mouth_center[0], hy - mouth_center[1]) <= MOUTH_COVER_DISTANCE:
            return True
    return False

# --- LOGIKA UTAMA ---
def main():
    # Placeholder untuk tampilan Streamlit
    frame_placeholder = st.empty()
    video_placeholder = st.empty()
    stop_button = st.button("Stop Program")

    # Load Video Kicau
    cat_cap = cv2.VideoCapture(CAT_VIDEO_PATH)
    
    # Setup Kamera
    cap = cv2.VideoCapture(0)
    
    wave_left = WaveDetector()
    wave_right = WaveDetector()
    last_mouth_cover_time = 0
    is_playing = False
    last_movement_time = 0

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        hands_result = hands_detector.process(rgb_frame)
        face_result = face_mesh_detector.process(rgb_frame)

        hand_centers = []
        mouth_center = None

        if face_result.multi_face_landmarks:
            mouth_center = get_mouth_center(face_result.multi_face_landmarks[0])
            cv2.circle(frame, (int(mouth_center[0]*w), int(mouth_center[1]*h)), 10, (255,255,255), -1)

        if hands_result.multi_hand_landmarks:
            for res in hands_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)
                hand_centers.append(get_hand_center(res))

        # Sort tangan kiri ke kanan
        hand_centers.sort(key=lambda p: p[0])
        now = time.time()

        # Deteksi Gerakan
        mouth_recent = (now - last_mouth_cover_time) <= COVER_WAVE_WINDOW
        if is_mouth_covered(hand_centers, mouth_center):
            last_mouth_cover_time = now

        wave_now = False
        if len(hand_centers) > 0:
            w_l = wave_left.update(hand_centers[0][0])
            w_r = wave_right.update(hand_centers[1][0]) if len(hand_centers) > 1 else False
            wave_now = w_l or w_r

        # Trigger Video
        if len(hand_centers) >= 2 and mouth_recent and wave_now and not is_playing:
            is_playing = True
            last_movement_time = now
            cat_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Logika Tampilan Video
        if is_playing:
            if wave_left.is_moving() or wave_right.is_moving():
                last_movement_time = now
            
            if now - last_movement_time > PLAY_TIMEOUT:
                is_playing = False
                video_placeholder.empty() # Hapus video dari layar
            else:
                ret_v, v_frame = cat_cap.read()
                if not ret_v:
                    cat_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_v, v_frame = cat_cap.read()
                if ret_v:
                    # Tampilkan video di bawah kamera
                    v_frame = cv2.cvtColor(v_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(v_frame, caption="🐾 KICAU MANIA AKTIF 🐾", use_container_width=True)

        # Update Frame Kamera
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    cap.release()
    cat_cap.release()

if __name__ == "__main__":
    main()