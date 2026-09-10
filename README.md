# SIBI Sign Language Recognition

Aplikasi desktop untuk mengenali gestur alfabet Bahasa Isyarat Indonesia (SIBI) A-Z menggunakan MediaPipe Hands dan model klasifikasi SVM. Project ini dibuat untuk kebutuhan Penulisan Ilmiah dengan alur lengkap: pengambilan dataset, training model, evaluasi, GUI real-time, dan packaging executable Windows.

## Fitur

- Deteksi landmark tangan secara real-time dengan MediaPipe.
- Klasifikasi alfabet SIBI A-Z menggunakan model SVM.
- GUI desktop berbasis CustomTkinter.
- Dropdown kamera dengan fallback otomatis ke kamera lain jika kamera terpilih gagal dibuka.
- Dukungan USB camera melalui konfigurasi `CAMERA_INDEX` atau pilihan kamera di GUI.
- Smoothing prediksi dan hold timing agar hasil tidak terlalu sensitif terhadap noise.
- Deteksi tambahan untuk huruf dinamis `J` dan `Z` berbasis pola gerak.
- Script pengumpulan dataset, training, evaluasi, pengecekan kamera, dan test otomatis.
- Build Windows executable menggunakan PyInstaller.

## Struktur Project

```text
.
|- main.py                         # Entry point GUI
|- requirements.txt                # Dependency Python
|- SIBI_Interpreter.spec           # Konfigurasi build PyInstaller
|- src/
|  |- config.py                    # Konfigurasi utama aplikasi
|  |- isyarat.py                   # Runtime recognizer dan OpenCV flow
|  |- isyarat_gui.py               # GUI desktop
|  |- sibi_core.py                 # Shared preprocessing landmark
|  `- app_logging.py               # Logging
|- scripts/
|  |- making_data.py               # Pengambilan dataset gesture
|  |- train.py                     # Training model SVM
|  |- evaluasi.py                  # Evaluasi dataset dan model
|  `- check_cameras.py             # Cek index kamera
|- data/
|  |- data_A.csv
|  |- ...
|  `- data_Z.csv
|- model/
|  |- mymodel.sav                  # Model SVM aktif
|  `- le.sav                       # Label encoder aktif
|- tests/
|  `- test_isyarat.py
`- docs/
   |- DATASET_MODEL_REPORT.md
   |- EVALUATION.md
   `- MANUAL_BOOK_SIBI_INTERPRETER.md
```

## Kebutuhan Sistem

- Windows 10/11
- Python 3.11.x
- Webcam laptop atau USB camera
- Pencahayaan ruangan yang cukup

## Instalasi Development

Buat dan aktifkan virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika environment sudah tersedia, gunakan langsung:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Menjalankan Aplikasi

Mode GUI utama:

```powershell
python main.py
```

Atau eksplisit melalui venv:

```powershell
.\.venv\Scripts\python.exe main.py
```

Mode OpenCV sederhana:

```powershell
python .\src\isyarat.py
```

## Menggunakan EXE Windows

Setelah build, executable berada di:

```text
dist\SIBI_Interpreter\SIBI_Interpreter.exe
```

Jalankan dari folder `dist\SIBI_Interpreter\`. Jangan memindahkan file `.exe` sendirian karena build PyInstaller menggunakan mode folder dan membutuhkan dependency di folder `_internal`.

Untuk membagikan aplikasi ke pengguna lain:

1. ZIP seluruh folder `dist\SIBI_Interpreter\`.
2. Upload ZIP ke Google Drive atau media lain.
3. Pengguna download dan extract ZIP.
4. Jalankan `SIBI_Interpreter.exe`.

## Build EXE

```powershell
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm .\SIBI_Interpreter.spec
```

Output build:

```text
dist\SIBI_Interpreter\SIBI_Interpreter.exe
```

## Cek Kamera

Untuk melihat index kamera yang tersedia:

```powershell
python .\scripts\check_cameras.py
```

Konfigurasi default ada di `src/config.py`:

```python
CAMERA_INDEX = 0
CAMERA_SCAN_INDICES = (2, 0, 1, 3)
```

Jika memakai USB camera, biasanya index kamera adalah `1` atau `2`. GUI juga menyediakan dropdown kamera dan fallback otomatis jika kamera terpilih gagal dibuka.

## Pengumpulan Dataset

Jalankan:

```powershell
python .\scripts\making_data.py
```

Alur:

1. Masukkan label huruf, misalnya `A`, `B`, atau `J`.
2. Kamera terbuka.
3. Tekan `Space` untuk mulai atau berhenti recording.
4. Tahan gesture stabil sampai counter bertambah.
5. Tekan `ESC` untuk keluar.

Dataset disimpan ke file:

```text
data\data_[LABEL].csv
```

Contoh untuk huruf `J`:

```text
data\data_J.csv
```

## Training Model

Setelah dataset diperbarui:

```powershell
python .\scripts\train.py
```

Output utama:

```text
model\mymodel.sav
model\le.sav
```

## Evaluasi

Validasi jumlah dataset per kelas:

```powershell
python .\scripts\evaluasi.py --validate-only --strict
```

Evaluasi holdout:

```powershell
python .\scripts\evaluasi.py --test-size 0.2 --seed 42 --no-in-sample
```

Evaluasi cross-validation:

```powershell
python .\scripts\evaluasi.py --cv 5 --seed 42 --no-in-sample
```

Ringkasan evaluasi terbaru tersedia di `docs/EVALUATION.md`.

## Testing

```powershell
python -m pytest -q
```

Status validasi terakhir:

```text
38 passed
```

## Troubleshooting

### Kamera tidak terbuka

- Pastikan kamera tidak sedang dipakai Zoom, browser, OBS, atau aplikasi lain.
- Jalankan `python .\scripts\check_cameras.py`.
- Ubah `CAMERA_INDEX` di `src/config.py` jika perlu.
- Untuk GUI, coba pilih kamera lain dari dropdown.

### EXE gagal load model

Pastikan folder `dist\SIBI_Interpreter\_internal\model\` berisi:

```text
mymodel.sav
le.sav
```

Jika model baru dibuat, rebuild EXE dengan PyInstaller.

### Hasil prediksi tidak stabil

- Gunakan background polos.
- Pastikan tangan masuk frame secara utuh.
- Gunakan pencahayaan merata.
- Jaga jarak tangan sekitar 30-50 cm dari kamera.
- Tahan gesture stabil sesuai durasi hold.

## Catatan Batasan

- Evaluasi dataset statis tidak sepenuhnya sama dengan performa real-time di kamera.
- Huruf dinamis seperti `J` dan `Z` dipengaruhi konsistensi gerakan.
- Akurasi dapat berubah jika pencahayaan, kamera, tangan pengguna, atau background berbeda jauh dari dataset training.

## Author

Ananta Raihan