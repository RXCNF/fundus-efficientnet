# Pengaruh Blok Atensi Tambahan pada EfficientNetB0 untuk Klasifikasi Citra Fundus

Kode dan hasil eksperimen untuk artikel *Pengaruh Blok Atensi Tambahan pada EfficientNetB0 untuk Klasifikasi Katarak, Glaukoma, dan Retinopati Diabetik pada Citra Fundus* (Muhammad Raihan Fasya dan Yuliani Puji Astuti, Universitas Negeri Surabaya).

Model dalam repositori ini **bukan alat diagnosis** dan belum divalidasi secara klinis.

## Ringkasan hasil

Akurasi pada 555 citra uji, rata-rata ± simpangan baku dari lima *seed*:

| Konfigurasi | Parameter | Akurasi (%) |
|---|---|---|
| EfficientNetB0 tanpa blok tambahan | 4.219.687 | 88,94 ± 0,82 |
| + SE (r = 4) | 5.040.487 | 88,29 ± 2,10 |
| + SE (r = 8) | 4.630.727 | 88,43 ± 2,20 |
| + SE (r = 16) | 4.425.847 | 88,25 ± 2,26 |
| + ECA | 4.219.692 | 88,22 ± 1,91 |
| + CBAM (r = 8) | 4.630.826 | 88,86 ± 1,82 |
| ResNet50 | 23.859.204 | 88,22 ± 0,61 |
| MobileNetV2 | 2.428.100 | 86,41 ± 1,68 |

Tidak ada blok atensi tambahan yang berbeda bermakna dari EfficientNetB0 tanpa blok tambahan.

Eksperimen tambahan (Bagian C, EfficientNetB0 tanpa blok tambahan kecuali disebutkan):

| Konfigurasi | *Seed* | Akurasi (%) |
|---|---|---|
| Tanpa augmentasi | 0–4 | 94,16 ± 0,41 |
| Tanpa augmentasi + SE (r = 8) | 0–4 | 93,62 ± 0,61 |
| Tanpa pemotongan acak | 0–2 | 92,43 ± 1,13 |
| Tanpa pembalikan horizontal | 0–2 | 89,13 ± 0,63 |
| Derau Gaussian 0,05 × 255 | 0–2 | 73,69 ± 11,58 |
| Resolusi 384 × 384 piksel | 0–2 | 87,39 ± 3,30 |
| Hanya tepi citra (kontrol jalan pintas) | 0–2 | 84,80 ± 1,91 |
| ViT-S/16 | 0–4 | 91,24 ± 1,37 |

Augmentasi yang dipakai pada studi ablasi menurunkan akurasi sekitar lima poin persentase, dan blok SE tetap tidak membantu tanpa augmentasi. Pengklasifikasi yang hanya memakai ukuran dan format berkas citra mencapai 70,27% pada data uji, yang menunjukkan kuatnya pengaruh sumber citra pada dataset ini.

## Isi repositori

| Folder | Isi |
|---|---|
| `notebook/bagian_A_model_utama.ipynb` | Satu model (EfficientNetB0 + SE r = 8, *seed* 123): pelatihan, metrik per kelas, selang kepercayaan, ROC, *precision-recall*, kalibrasi, ambang glaukoma, Grad-CAM, waktu inferensi |
| `notebook/bagian_B_ablasi.ipynb` | Studi ablasi dan model pembanding: 8 konfigurasi × 5 *seed*, dua pelatihan berjalan bersamaan pada GPU T4 x2 |
| `data/pembagian_70_15_15.csv` | Daftar berkas pada data latih, validasi, dan uji (2.648 / 560 / 555) beserta kelompok citra mirip |
| `data/pasangan_citra_mirip.csv` | 939 pasangan citra dengan jarak Hamming pHash ≤ 4 dan selisih piksel rata-ratanya |
| `notebook/bagian_C_eksperimen_tambahan.ipynb`, `notebook/bagian_C_sesi2.ipynb` | Eksperimen tambahan: pengaruh augmentasi, resolusi 384 × 384, ViT-S/16, kontrol tepi citra, dan SE tanpa augmentasi (sesi 2 melanjutkan sesi 1) |
| `analisis/` | Skrip analisis lokal: kontrol metadata dan kualitas citra, analisis kesalahan (konsistensi antar-*seed*, keyakinan), pemeriksaan salinan cermin/terputar di dalam dataset beserta analisis sensitivitasnya, dan pengukuran TensorFlow Lite. Jalur berkas di dalam skrip mengikuti susunan folder keluaran notebook dan perlu disesuaikan; pengukuran TensorFlow Lite memerlukan bobot model hasil Bagian A |
| `hasil/bagian_C/` | Metrik 30 pelatihan Bagian C dan tabel ringkasannya |
| `hasil/analisis_kesalahan_A15.json`, `hasil/konsistensi_kesalahan_per_citra.csv` | Hasil analisis kesalahan dan jumlah pelatihan yang salah per citra uji |
| `hasil/cek_salinan_augmentasi.csv`, `hasil/citra_uji_bersalinan.csv` | Pasangan citra yang mirip setelah dicerminkan atau diputar 180° (dengan selisih piksel; < 3 dianggap salinan) dan 61 citra uji yang memiliki salinan di data latih atau validasi |
| `hasil/ablasi/` | Metrik 40 pelatihan dan probabilitas keluaran per citra uji (`probabilitas_uji/*.npy`, urutan baris sama dengan baris `subset == uji` pada berkas pembagian data) |
| `hasil/model_utama/` | Prediksi per citra, riwayat pelatihan, tabel, dan gambar model utama |

Bobot model tidak disertakan karena ukurannya (53 MB); bobot dapat dihasilkan ulang dengan menjalankan notebook.

## Data

Dataset: [Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) (Kaggle, 4.217 citra, empat kelas). Citra **tidak** disertakan di repositori ini; basis datanya berlisensi Open Database License (ODbL) v1.0 dan hak cipta citra berada pada pemilik aslinya. Nama berkas pada folder `data/` dan `hasil/` ditulis sebagai `kelas/nama_berkas`.

Sebanyak 454 citra dalam 71 kelompok citra mirip yang berlabel saling bertentangan dikeluarkan, sehingga tersisa 3.763 citra. Pembagian 70:15:15 dilakukan pada tingkat kelompok dengan *seed* 123.

## Cara menjalankan

Eksperimen dijalankan di Kaggle (Python 3.13, TensorFlow 2.21.0 dengan `tf-keras` 2.21.0, GPU NVIDIA Tesla T4).

1. Buat notebook Kaggle, impor salah satu berkas di folder `notebook/`.
2. Tambahkan dataset *eye-diseases-classification* sebagai Input, pilih Accelerator **GPU T4 x2**, dan aktifkan Internet (untuk mengunduh bobot ImageNet).
3. Jalankan lewat **Save Version → Save & Run All**. Keluaran tersimpan di `/kaggle/working/`.

Bagian A memerlukan sekitar 1 jam. Bagian C memerlukan sekitar 26 jam GPU untuk sesi 1 dan 14 jam GPU untuk sesi 2 (hasil sesi 1 sudah tertanam di notebook sesi 2). Bagian B memerlukan sekitar 18 jam GPU dan berhenti sendiri sebelum batas sesi 12 jam; untuk melanjutkan, tambahkan keluaran sesi sebelumnya sebagai Input lalu jalankan lagi, dan pelatihan yang sudah selesai dilewati.

## Lisensi

Kode pada repositori ini berlisensi MIT. Lisensi tersebut tidak mencakup dataset.
