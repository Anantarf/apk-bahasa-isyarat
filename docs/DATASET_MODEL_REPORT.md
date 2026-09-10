# Dataset and Model Report

Dokumen ini merangkum kondisi dataset dan model aktif pada project SIBI Sign Language Recognition.

## Ringkasan Dataset

- Lokasi dataset: `data/`
- Format file: `data_[LABEL].csv`
- Label: alfabet `A-Z`
- Jumlah fitur per sampel: 210
- Target minimal per kelas: 300 sampel
- Total sampel terbaru: 12.729
- Min/Max sampel per kelas: 321 / 1.635
- Imbalance ratio terbaru: 5.09x

Semua kelas sudah mencapai target minimal 300 sampel.

## Jumlah Sampel Per Kelas

```text
A: 875   B: 1182  C: 1635  D: 655   E: 634   F: 775
G: 366   H: 345   I: 367   J: 321   K: 385   L: 446
M: 391   N: 328   O: 330   P: 338   Q: 336   R: 341
S: 331   T: 335   U: 343   V: 349   W: 343   X: 328
Y: 326   Z: 324
```

## Status Validasi

Command:

```powershell
python .\scripts\evaluasi.py --validate-only --strict
```

Hasil:

```text
Semua kelas OK dan memenuhi TARGET_PER_CLASS.
```

## Model Aktif

Model aktif berada di:

```text
model\mymodel.sav
model\le.sav
```

Model menggunakan pendekatan klasifikasi SVM dari fitur landmark tangan MediaPipe. `le.sav` dipakai sebagai sumber label runtime agar urutan label tetap konsisten antara training dan inferensi.

## Hasil Evaluasi Ringkas

Holdout stratified split dengan `test_size=0.2` dan `seed=42`:

```text
Accuracy: 0.9855
Macro avg  (Precision / Recall / F1): 0.9920 / 0.9915 / 0.9917
Weighted   (Precision / Recall / F1): 0.9857 / 0.9855 / 0.9855
```

Output training terbaru menunjukkan:

```text
Accuracy: 0.9904405652535329
```

## Analisis Error

Confusion terbesar pada evaluasi holdout terbaru:

```text
C -> B : 19
B -> C : 9
W -> V : 2
```

Huruf yang masih perlu perhatian paling besar adalah `B` dan `C`, karena bentuk gesture keduanya relatif mudah tertukar jika jari atau telapak tangan tidak terlihat jelas.

## Rekomendasi Pengembangan Dataset

- Tambah variasi data untuk `B` dan `C`.
- Gunakan background polos dan cahaya merata.
- Rekam dari jarak dekat dan sedang.
- Hindari sampel blur atau tangan keluar frame.
- Untuk `J` dan `Z`, rekam gerakan yang konsisten dengan tempo stabil.

## Catatan Batasan

Evaluasi ini berbasis dataset fitur statis. Performa real-time di kamera bisa berbeda karena dipengaruhi kamera, cahaya, background, posisi tangan, dan konsistensi gesture pengguna.