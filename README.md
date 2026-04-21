# Kicau Mania Detector

Kicau Mania Detector merupakan mini project berbasis *computer vision* yang dirancang untuk mendeteksi gesture secara real-time melalui kamera. Sistem ini merespons kombinasi gerakan tertentu dengan menjalankan aksi berupa membuka video MP4 pada tab baru.

Gesture yang digunakan sebagai pemicu terdiri dari:

* Deteksi dua tangan
* Posisi tangan menutupi area mulut
* Gerakan mengibas tangan ke arah kiri dan kanan

---

## Requirements

Sebelum menjalankan aplikasi, pastikan lingkungan telah memenuhi kebutuhan berikut:

* Perangkat dengan kamera (webcam)
* Python versi 3.9 – 3.11 (direkomendasikan 3.10)
* Koneksi internet untuk instalasi dependensi

---

## Instalasi dan Setup

### 1. Install Python

Unduh Python melalui situs resmi:
[https://www.python.org/downloads/](https://www.python.org/downloads/)

Pastikan opsi "Add Python to PATH" dicentang saat proses instalasi.

---

### 2. Akses Terminal

* Windows: `Win + R` → ketik `cmd`
* macOS: buka `Terminal`

---

### 3. Masuk ke Direktori Project

```bash
cd path/ke/folder/kicau_mania
```

Contoh:

```bash
cd C:\Users\Username\Downloads\kicau_mania
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Persiapan File Video

* Gunakan file video berformat `.mp4`
* Disarankan menggunakan nama `kicau-mania.mp4`
* Letakkan file pada direktori yang sama dengan `main.py`

Struktur direktori:

```
kicau_mania/
├── kicau.py
├── requirements.txt
├── README.md
└── kicau-mania.mp4
```

---

### 6. Menjalankan Aplikasi

```bash
python kicau.py
```

---

## Cara Penggunaan

| Kondisi Gesture                                                      | Output                                |
| -------------------------------------------------------------------- | ------------------------------------- |
| Dua tangan terdeteksi, menutup mulut, dan melakukan gerakan mengibas | Video MP4 terbuka pada tab baru       |
| Gesture belum memenuhi kriteria                                      | Status deteksi ditampilkan pada layar |
| Tekan tombol Q                                                       | Keluar dari aplikasi                  |

---

## Konfigurasi

Parameter sistem dapat disesuaikan melalui bagian konfigurasi pada file `main.py`:

```python
CAT_SIZE       = (220, 220)
WAVE_THRESHOLD = 2
MOVE_MINIMUM   = 0.018
MOUTH_COVER_DISTANCE = 0.16
TRIGGER_COOLDOWN = 4.0
```

Keterangan:

* WAVE_THRESHOLD: jumlah perubahan arah gerakan untuk mendeteksi kibasan
* MOVE_MINIMUM: tingkat sensitivitas pergerakan
* MOUTH_COVER_DISTANCE: jarak maksimum tangan terhadap mulut
* TRIGGER_COOLDOWN: jeda waktu antar trigger

---

## Troubleshooting

| Permasalahan             | Solusi                                                  |
| ------------------------ | ------------------------------------------------------- |
| ModuleNotFoundError      | Jalankan ulang `pip install -r requirements.txt`        |
| Kamera tidak terdeteksi  | Ubah indeks kamera pada `cv2.VideoCapture()`            |
| Video tidak terbuka      | Pastikan file `.mp4` tersedia dan browser default aktif |
| Gesture sulit terdeteksi | Sesuaikan nilai `MOVE_MINIMUM`                          |
| Trigger terlalu sering   | Tingkatkan `WAVE_THRESHOLD` atau `TRIGGER_COOLDOWN`     |

---

## Mekanisme Sistem

Alur kerja aplikasi secara umum:

```
Input kamera
→ Deteksi tangan (MediaPipe Hands)
→ Deteksi wajah (Face Mesh)
→ Evaluasi gesture
→ Trigger aksi (membuka video)
```

---

## Dependencies

Library utama yang digunakan:

* OpenCV
* MediaPipe
* MediaPipe Face Mesh
* NumPy
* Pillow (opsional)
