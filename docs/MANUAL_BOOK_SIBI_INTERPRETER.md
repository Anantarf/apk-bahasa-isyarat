# Manual Book SIBI Interpreter

Manual ini menjelaskan cara menjalankan aplikasi SIBI Interpreter versi desktop Windows.

## 1. Menjalankan Aplikasi dari EXE

Gunakan folder hasil build:

```text
dist\SIBI_Interpreter\
```

Di dalam folder tersebut, jalankan:

```text
SIBI_Interpreter.exe
```

Jangan memindahkan file `.exe` sendirian karena aplikasi membutuhkan folder `_internal` yang berisi dependency, model, dan file pendukung.

## 2. Membagikan Aplikasi

Untuk membagikan aplikasi lewat Google Drive:

1. Compress seluruh folder `dist\SIBI_Interpreter\` menjadi ZIP.
2. Upload ZIP ke Google Drive.
3. Pengguna download ZIP.
4. Extract ZIP.
5. Jalankan `SIBI_Interpreter.exe`.

## 3. Tampilan Utama

Aplikasi memiliki beberapa bagian utama:

- Area kamera untuk menampilkan preview webcam.
- Dropdown kamera untuk memilih kamera laptop atau USB camera.
- Tombol mulai/hentikan deteksi.
- Panel hasil terjemahan untuk menampilkan huruf yang berhasil dikenali.
- Tombol reset untuk menghapus hasil.
- Tombol salin teks untuk menyalin hasil ke clipboard.

## 4. Cara Menggunakan

1. Pastikan kamera sudah terhubung.
2. Buka `SIBI_Interpreter.exe`.
3. Pilih kamera dari dropdown jika diperlukan.
4. Klik `MULAI DETEKSI`.
5. Arahkan tangan ke kamera.
6. Tahan gesture selama kurang lebih 2.5 detik.
7. Huruf hasil deteksi akan muncul di panel hasil.
8. Gunakan `RESET` untuk menghapus teks dan mulai ulang.

## 5. Shortcut Keyboard

```text
Space : Mulai atau hentikan deteksi
R     : Reset hasil teks
Esc   : Keluar dari aplikasi
```

## 6. Tips Deteksi Yang Baik

- Gunakan pencahayaan yang cukup dan merata.
- Hindari background yang terlalu ramai.
- Pastikan tangan masuk frame secara utuh.
- Jaga jarak tangan sekitar 30-50 cm dari kamera.
- Tahan gesture dengan stabil.
- Untuk huruf `J` dan `Z`, lakukan gerakan dengan tempo stabil dan tidak keluar frame.

## 7. Menggunakan USB Camera

1. Colok USB camera sebelum membuka aplikasi.
2. Buka aplikasi.
3. Pilih kamera USB dari dropdown.
4. Klik `MULAI DETEKSI`.

Jika kamera terpilih gagal dibuka, aplikasi akan mencoba fallback ke kamera lain yang tersedia.

## 8. Troubleshooting

### Kamera tidak muncul

- Pastikan kamera tidak digunakan aplikasi lain.
- Cabut dan colok ulang USB camera.
- Tutup lalu buka ulang aplikasi.
- Coba pilih kamera lain dari dropdown.

### Aplikasi gagal load model

Pastikan folder `_internal\model\` di dalam folder build berisi:

```text
mymodel.sav
le.sav
```

Jika model baru dibuat dari training ulang, EXE harus dibuild ulang dengan PyInstaller.

### Windows menampilkan peringatan keamanan

EXE hasil PyInstaller bisa memunculkan warning dari Windows Defender atau SmartScreen karena aplikasi belum ditandatangani sertifikat digital. Jalankan hanya jika file berasal dari build project ini.

## 9. Batasan

- Hasil real-time dipengaruhi kualitas kamera, cahaya, background, dan konsistensi gesture.
- Huruf dengan bentuk mirip, seperti `B/C`, `U/V`, atau `L/Y`, lebih mudah tertukar jika pose tidak jelas.
- Evaluasi model berbasis dataset statis tidak selalu sama dengan hasil kamera real-time.