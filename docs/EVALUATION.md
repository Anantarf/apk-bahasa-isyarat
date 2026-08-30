# Evaluasi Model (Precision, Recall, F1)

Dokumen ini menjelaskan cara menjalankan evaluasi dan cara membaca output dari `scripts/evaluasi.py`.

## Yang Dievaluasi

- Yang dievaluasi adalah **model klasifikasi SVM** (file: `model/mymodel.sav`) terhadap **dataset fitur** di `data/data_*.csv`.
- Dataset ini berisi **fitur numerik** (210 angka per sampel). Jadi evaluasi ini mengukur performa **model SVM pada fitur statis**.

> Catatan: fitur runtime aplikasi (mis. smoothing vote, progress/hold timing, dan deteksi motion J/Z) adalah layer real-time yang bekerja di webcam dan **tidak termasuk** evaluasi dataset statis ini.

## Cara Menjalankan

Gunakan python dari venv project:

```powershell
python scripts/evaluasi.py --cv 5 --seed 42 --no-in-sample
```

Quality gate sebelum retrain (strict):

```powershell
python scripts/evaluasi.py --validate-only --strict
```

Jika ada kelas dengan jumlah data di bawah `TARGET_PER_CLASS` (lihat `src/config.py`), command ini akan gagal (exit code != 0).

Opsional (holdout split saja):

```powershell
python scripts/evaluasi.py --test-size 0.2 --seed 42
```

## Cara Membaca Output

### 1) Dataset summary

Bagian ini menampilkan:

- Total sampel
- Jumlah fitur per sampel (seharusnya 210)
- Jumlah kelas (A–Z)
- **Support per kelas** = jumlah data pada kelas tersebut

Jika support timpang (kelas tertentu jauh lebih banyak), maka metrik agregat perlu dibaca dengan hati-hati.

### 2) Precision, Recall, F1 (per kelas)

Untuk suatu kelas (mis. huruf `J`):

- **Precision(J)**: dari semua prediksi yang diklaim sebagai `J`, berapa yang benar `J`.
- **Recall(J)**: dari semua data yang benar-benar `J`, berapa yang berhasil terdeteksi sebagai `J`.
- **F1(J)**: ringkasan Precision dan Recall (harmonic mean).

### 3) Macro vs Weighted

Karena dataset **tidak seimbang**, dua angka ini penting:

- **Macro avg (F1)**: rata-rata F1 semua kelas dengan bobot sama.
  - Cocok untuk menilai performa “adil” untuk setiap huruf.
- **Weighted (F1)**: rata-rata F1 dengan bobot sesuai jumlah data per kelas.
  - Akan lebih dipengaruhi kelas yang datanya banyak.

Rekomendasi untuk penulisan ilmiah:

- Utamakan **Macro-F1** sebagai metrik utama untuk dataset tidak seimbang.
- Sertakan **Weighted-F1** sebagai pelengkap.
- Sertakan juga tabel per kelas (Precision/Recall/F1/Support) dan confusion matrix.

## Hasil Evaluasi (Seed=42)

Hasil berikut adalah output yang sudah dihasilkan di workspace ini.

### A) Holdout (Stratified split, test_size=0.2)

- Accuracy: **0.9803**
- Macro avg (Precision / Recall / F1): **0.9884 / 0.9853 / 0.9860**
- Weighted (Precision / Recall / F1): **0.9810 / 0.9803 / 0.9803**

### B) Cross-validation 5-Fold (StratifiedKFold, cv=5)

- Accuracy: **0.9782**
- Macro avg (Precision / Recall / F1): **0.9816 / 0.9766 / 0.9787**
- Weighted (Precision / Recall / F1): **0.9786 / 0.9782 / 0.9783**

> Output `scripts/evaluasi.py` juga mencetak confusion matrix (rows=true, cols=pred) untuk analisis kesalahan.

## Catatan Reproducibility (Versi Library)

Model `mymodel.sav` dibuat dengan scikit-learn versi lama. Untuk kerapihan ilmiah dan konsistensi, environment project ini dipin:

- `scikit-learn==1.3.2`
- `numpy==1.26.4` (kompatibel dengan scikit-learn 1.3.2)

Versi ini memastikan load model tidak memunculkan warning mismatch.
