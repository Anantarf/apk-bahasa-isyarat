# SIBI Sign Language Recognition

Aplikasi penerjemah Bahasa Isyarat SIBI (A-Z) berbasis Computer Vision dan Machine Learning.
Pipeline utama: ambil data -> train model -> evaluasi -> inferensi real-time.

## Fitur Utama

- Deteksi tangan real-time dengan MediaPipe.
- Klasifikasi gesture huruf A-Z menggunakan SVM.
- Stabilitas prediksi dengan hold gesture + smoothing.
- Deteksi huruf dinamis J/Z berbasis pola gerak ujung jari telunjuk.
- Konfigurasi terpusat di `config.py`.
- GUI desktop opsional (`isyarat_gui.py`) selain mode OpenCV (`isyarat.py`).
- Utilities training, evaluasi, dan testing (`pytest`).

## Quick Start

### 1. Install dependencies

Disarankan memakai environment di `library/`.

```powershell
& .\library\Scripts\python.exe -m pip install -r .\requirements.txt
```

### 2. Jalankan inferensi real-time (OpenCV window)

```powershell
& .\library\Scripts\python.exe .\isyarat.py
```

Kontrol default:

- `ESC`: keluar aplikasi.

## Workflow End-to-End

### 1. Kumpulkan data gesture

```powershell
& .\library\Scripts\python.exe .\making_data.py
```

Saat pengambilan data:

- Input label saat diminta (contoh: `A`, `B`, ...).
- `Space`: start/stop recording.
- `ESC`: keluar.

### 2. Train model

```powershell
& .\library\Scripts\python.exe .\train.py
```

Output artifact utama di folder `model/`:

- `mymodel.sav` (model SVM)
- `le.sav` (label encoder, source-of-truth label runtime)

### 3. Evaluasi dataset dan model

```powershell
& .\library\Scripts\python.exe .\evaluasi.py --validate-only
& .\library\Scripts\python.exe .\evaluasi.py --test-size 0.2 --seed 42
& .\library\Scripts\python.exe .\evaluasi.py --cv 5 --seed 42
```

Laporan tambahan: `EVALUATION.md`.

### 4. Jalankan GUI (opsional)

```powershell
& .\library\Scripts\python.exe .\isyarat_gui.py
```

## Struktur Project

```text
.
|- config.py
|- making_data.py
|- train.py
|- evaluasi.py
|- EVALUATION.md
|- isyarat.py
|- isyarat_gui.py
|- test_isyarat.py
|- test_cameras.py
|- requirements.txt
|- data/
|  |- data_A.csv
|  |- ...
|  `- data_Z.csv
`- model/
   |- mymodel.sav
   |- le.sav
   |- mymodel_balanced.sav
   `- mymodel_weighted.sav
```

## Konfigurasi Penting

Edit `config.py` sesuai kebutuhan.

```python
# Timing
GESTURE_DURATION: float = 2.5
DISPLAY_DURATION: float = 999999.0
PREDICTION_DELAY: float = 2.5
RAW_PREDICTION_INTERVAL: float = 0.1

# Camera
CAMERA_INDEX: int = 0
CAMERA_FLIP_HORIZONTAL: bool = True

# Hand selection
RIGHT_HAND_ONLY: bool = True
TARGET_HAND_LABEL: str = "Left"

# Smoothing
PREDICTION_SMOOTHING_WINDOW: int = 3

# Motion letters (J/Z)
ENABLE_JZ_MOTION: bool = True
JZ_WINDOW_SECONDS: float = 0.6
JZ_MIN_MOTION_LEVEL: str = "medium"
```

Catatan:

- `DISPLAY_DURATION` besar untuk mempertahankan teks output lebih lama.
- Runtime membaca label dari `model/le.sav` jika tersedia.

## Testing

```powershell
& .\library\Scripts\python.exe -m pytest -v
```

## Troubleshooting

### Kamera tidak terbuka

- Ubah `CAMERA_INDEX` di `config.py` (coba `0`, `1`, `2`).
- Pastikan kamera tidak dipakai aplikasi lain.

### Model/label artifact tidak ditemukan

- Pastikan folder `model/` berisi minimal `mymodel.sav` dan `le.sav`.
- Jika belum ada, jalankan `train.py` terlebih dahulu.

### Akurasi rendah

- Gunakan pencahayaan merata.
- Hindari background ramai.
- Jaga jarak tangan sekitar 30-50 cm.
- Pastikan tangan masuk frame secara utuh.
- Tahan gesture stabil sesuai `GESTURE_DURATION`.

### Import error di VS Code (Pylance)

Jika muncul warning seperti `Import 'numpy/cv2/mediapipe' could not be resolved`, biasanya interpreter belum menunjuk ke environment project.

- `Ctrl+Shift+P` -> `Python: Select Interpreter` -> pilih `...\library\Scripts\python.exe`
- Install ulang dependencies:

```powershell
& .\library\Scripts\python.exe -m pip install -r .\requirements.txt
```

## Author

Ananta Raihan
