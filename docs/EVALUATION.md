# Evaluasi Model SIBI

Dokumen ini merangkum cara evaluasi dataset dan model klasifikasi alfabet SIBI A-Z pada project ini.

## Lingkup Evaluasi

Evaluasi dilakukan terhadap:

- Dataset fitur landmark di `data/data_*.csv`.
- Model SVM aktif di `model/mymodel.sav`.
- Label encoder aktif di `model/le.sav`.

Setiap sampel berisi 210 fitur numerik, yaitu sequence 5 frame x 42 koordinat landmark tangan. Evaluasi ini mengukur performa model pada dataset fitur statis. Flow real-time seperti smoothing, hold timing, camera fallback, dan deteksi gerak `J/Z` tetap perlu diuji manual melalui kamera.

## Command Evaluasi

Validasi dataset minimal 300 sampel per kelas:

```powershell
python .\scripts\evaluasi.py --validate-only --strict
```

Evaluasi holdout stratified split:

```powershell
python .\scripts\evaluasi.py --test-size 0.2 --seed 42 --no-in-sample
```

Evaluasi cross-validation 5-fold:

```powershell
python .\scripts\evaluasi.py --cv 5 --seed 42 --no-in-sample
```

## Status Dataset Terbaru

Target minimal per kelas: 300 sampel.

```text
A: 875   B: 1182  C: 1635  D: 655   E: 634   F: 775
G: 366   H: 345   I: 367   J: 321   K: 385   L: 446
M: 391   N: 328   O: 330   P: 338   Q: 336   R: 341
S: 331   T: 335   U: 343   V: 349   W: 343   X: 328
Y: 326   Z: 324
```

Semua kelas sudah mencapai target minimal. Imbalance ratio terbaru turun menjadi 5.09x dari kondisi sebelumnya yang jauh lebih timpang.

## Hasil Holdout Terbaru

Command:

```powershell
python .\scripts\evaluasi.py --test-size 0.2 --seed 42 --no-in-sample
```

Hasil:

```text
Accuracy: 0.9855
Macro avg  (Precision / Recall / F1): 0.9920 / 0.9915 / 0.9917
Weighted   (Precision / Recall / F1): 0.9857 / 0.9855 / 0.9855
```

Confusion terbesar:

```text
C -> B : 19
B -> C : 9
W -> V : 2
V -> L : 1
T -> M : 1
T -> C : 1
S -> V : 1
M -> G : 1
M -> A : 1
L -> Y : 1
```

## Hasil Training Terbaru

Output `scripts/train.py` setelah dataset diperbarui menunjukkan:

```text
Accuracy: 0.9904405652535329
```

Huruf dengan akurasi per kelas terendah pada output training:

```text
B: 94.9%
C: 95.5%
L: 98.9%
R: 98.5%
X: 98.5%
```

Artinya, peningkatan berikutnya sebaiknya tetap difokuskan pada pasangan huruf yang mudah tertukar, terutama `B` dan `C`.

## Cara Membaca Metrik

- Precision: dari semua prediksi suatu huruf, berapa yang benar.
- Recall: dari semua data asli suatu huruf, berapa yang berhasil dikenali.
- F1-score: ringkasan precision dan recall.
- Macro average: rata-rata semua kelas dengan bobot sama.
- Weighted average: rata-rata dengan bobot mengikuti jumlah sampel per kelas.

Karena dataset tidak sepenuhnya seimbang, Macro-F1 perlu dilaporkan bersama accuracy.

## Rekomendasi Lanjutan

- Tambah data variasi untuk `B` dan `C`.
- Rekam ulang sampel yang blur, terlalu miring, atau gesture-nya tidak konsisten.
- Uji real-time dengan kamera laptop dan USB camera.
- Setelah model diperbarui, rebuild EXE agar model baru ikut masuk ke bundle.

## Catatan Reproducibility

Dependency utama yang dipakai:

```text
Python 3.11.x
scikit-learn==1.3.2
numpy==1.26.4
mediapipe==0.10.14
opencv-python==4.10.0.84
PyInstaller==6.9.0
```