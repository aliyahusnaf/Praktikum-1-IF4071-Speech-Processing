# Pengenalan Vowel Bahasa Indonesia

Praktikum IF4071: pengenalan vowel a, i, u, e, o dengan DTW dan Model Gaussian.

## Dataset

6 speaker, masing-masing 5 vowel dengan 2 rekaman: total 60 audio.
Format audio: WAV PCM 16-bit, mono, 16000 Hz.

- `data/raw/`: rekaman asli.
- `data/processed/template/`: 30 audio rekaman pertama.
- `data/processed/test/closed/`: 30 audio rekaman kedua.
- `data/raw/open/`: rekaman open speaker dari kelompok lain (G03/Kel3, G04, G07, G09),
  disimpan sesuai struktur asli `A/ E/ I/ O/ U/`.
- `data/processed/test/open/`: 80 audio open speaker (16 speaker x 5 vowel).

Nama file: `S1_a1.wav` = speaker S1, vowel a, rekaman 1.
Open set: `G04S03_a.wav` = kelompok 4, speaker S03, vowel a (hanya satu rekaman).
Speaker: S1 Aliya, S2 Farrel, S3 Fithra, S4 Kefas, S5 Rakdaf, S6 Timur.

## Menjalankan program

Jalankan dari folder proyek dengan Python 3.9+:

```powershell
python -m pip install -r requirements.txt
python -m src.preprocessing
python -m unittest discover -s tests -v
```

### Preprocessing dan data loading

Untuk membuat ulang WAV langsung dari raw (FFmpeg harus tersedia):

```powershell
python -m src.prepare_dataset --ffmpeg ffmpeg
```

Perintah ini mengganti 60 WAV processed dan menyimpan backup di outputs.
Setiap channel didekode terpisah, tanpa merata-ratakan kiri dan kanan.
Channel dipilih konsisten per speaker dari median kontras energi frame,
dengan penalti untuk clipping. Ini perkiraan, bukan pengukuran SNR atau jaminan
channel paling bersih. Hasil saat ini: S3 kanan, speaker lain channel pertama
(kiri untuk stereo). Setelah itu dilakukan DC offset removal dan trimming.
Laporan pilihan channel, durasi, dan pesan decoder ada di
`outputs/preprocessing_report.json`. File raw tidak diubah.
FFmpeg masih mengeluarkan pesan decoding ALAC pada raw S4; WAV hasilnya lolos
validasi, tetapi kualitas suaranya perlu dicek dengan mendengarkan.

`src/preprocessing.py` mengecek format audio, sinyal kosong/rusak, jumlah
rekaman, kelengkapan vowel, dan duplikat. Hasil validasi tampil sebagai JSON.
`src/data_loader.py` membaca audio sebagai array NumPy float32 satu dimensi.
Skala sampel int16 dibagi 32768, jadi rentangnya [-1, 1).

```python
from src.data_loader import load_dataset

data = load_dataset()
templates = data["template"]
closed_speakers = data["closed"]
closed_recordings = templates  # Pakai lagi audio template.

record = templates[0]
audio = record["audio"]
sample_rate = record["sample_rate"]
print(record["filename"], record["speaker"], record["label"])
print(audio.shape, audio.dtype, sample_rate)
# features = extract_mfcc_39(audio, sample_rate)
```

Setiap record berisi `audio`, `sample_rate`, `speaker`, `label`, `recording`,
`filename`, dan `path`. Bisa juga baca satu file dengan `load_audio(path)`.
Path dataset default mengikuti lokasi proyek. Untuk lokasi lain:
`load_dataset(data_dir="C:/lokasi/data/processed")`.
Loader menolak dataset yang gagal validasi. WAV di `data/processed/` sudah
melalui penghilangan DC offset dan trimming jeda awal/akhir. Jadi `load_dataset()`
langsung membaca hasilnya tanpa trimming ulang. File raw tetap utuh.

Trimming memakai energi RMS: frame 20 ms, hop 10 ms, ambang -30 dB dari RMS
tertinggi (minimal 0.0001), bagian aktif minimal 5 frame, dan padding 80 ms.
Jeda di tengah tetap disimpan. Tidak ada normalisasi volume atau noise reduction.
Saat disimpan, hasil dibulatkan kembali ke PCM 16-bit dan dibatasi ke rentang int16.

Durasi sebelum/sesudah ada di `outputs/preprocessing_report.json`.
Salinan WAV sebelum trimming ada di `outputs/processed_before_trim.zip`.
Backup sebelum perbaikan channel memakai nama `processed_before_channel_fix_*.zip`.
Hasil perlu didengarkan untuk memastikan bagian vowel tidak terpotong.
Kalau ingin mencoba preprocessing pada audio lain, pakai `preprocess_audio()`
atau `load_audio(path, preprocess=True)`. Jangan aktifkan lagi untuk dataset
processed yang sekarang supaya tidak terpotong berulang.

Konversi M4A sebelumnya dilakukan dari folder raw masing-masing speaker:

```powershell
mkdir wav
Get-ChildItem -Filter *.m4a -File | ForEach-Object {
    ffmpeg -n -i $_.FullName -ar 16000 -ac 1 -c:a pcm_s16le ".\wav\$($_.BaseName).wav"
}
```

WAV S1 juga sudah diubah dari 22050 ke 16000 Hz dengan opsi FFmpeg yang sama,
langsung dari WAV di `data/raw/aliya/`. File raw tetap disimpan.

### Ekstraksi MFCC

`src/mfcc.py` memakai `python_speech_features` dengan parameter: frame 25 ms, hop 10 ms, pre-emphasis 0.97, jendela Hamming, NFFT 512, 26 filter mel, log, DCT, lalu c0 diganti log energi frame. Lifter dimatikan
karena tidak ada di materi. Delta dan delta-delta memakai 2 frame maju/mundur.

```python
from src.data_loader import load_dataset
from src.mfcc import extract_features, extract_mfcc_39, FEATURE_NAMES

data = load_dataset()
templates = extract_features(data["template"])  # tiap record dapat key "mfcc"
features = templates[0]["mfcc"]                  # shape (jumlah_frame, 39)
# Satu audio saja: extract_mfcc_39(audio, sample_rate)
```

`FEATURE_NAMES` adalah list 39 nama kolom, sesuai urutan kolom di matriks `mfcc`.
Isinya hanya label, tidak dipakai dalam perhitungan, misalnya untuk tahu
kolom ke-0 adalah energi atau untuk judul grafik. Urutannya `energy, c1..c12`
(fitur statis), lalu `d_energy, d_c1..d_c12` (delta), lalu
`dd_energy, dd_c1..dd_c12` (delta-delta).

CMVN dibuat opsional dengan default mati( bisa diatur di `extract_features(records, cmvn=True)`).
Statistiknya per speaker, diambil dari semua rekaman speaker itu di `records` agar yang hilang adalah suara rata-rata dari speaker.
Tidak pakai CMVN per ucapan karena vowel diucapkan sendirian dan stasioner, sehingga rata-ratanya adalah ciri vowel itu sendiri.
```powershell
python -m src.mfcc                                   # ringkasan shape fitur
python -m src.mfcc --cmvn --output outputs/mfcc.npz  # simpan fitur
```

### Data uji open speaker

```powershell
python -m src.prepare_open_set
```

Membaca WAV di `data/raw/open/`, menyeragamkan nama (`Kel3_S01_A.wav` dan
`G09_S01_A01.wav` menjadi `G03S01_a.wav` dan `G09S01_a.wav`), lalu memakai
preprocessing yang sama dengan data closed: hapus DC offset dan trimming jeda awal/akhir.
Semua file sumber sudah PCM 16-bit mono 16000 Hz, jadi tidak perlu konversi.
Laporan durasi ada di `outputs/preprocessing_open_report.json`.

```python
from src.data_loader import load_open_set
open_records = load_open_set()  # 80 record, recording = None
```

### Model Gaussian (Sistem 2)

`src/gaussian.py`. Satu Gaussian multivariat per vowel:

1. Latih: semua frame MFCC 39 dimensi dari 6 rekaman template vowel itu digabung,
   lalu dihitung mean (39) dan covariance (39 x 39). Diagonal covariance ditambah
   `1e-4 x rata-rata varians` supaya selalu bisa diinvers.
2. Uji: tiap frame dihitung
   `NLL = 0.5 (D log 2pi + log|cov| + (x-mean)^T cov^-1 (x-mean))`, lalu dirata-rata
   per frame. Model dengan NLL terkecil dipilih sebagai prediksi.

Log determinan dan jarak Mahalanobis dihitung lewat dekomposisi Cholesky (NumPy).

```powershell
python -m src.gaussian              # konfigurasi utama: full covariance, tanpa CMVN
python -m src.gaussian --cmvn --covariance diag
python -m src.gaussian --ablation   # bandingkan full/diag x CMVN on/off
```

Hasil disimpan ke `outputs/gaussian_results.json` (akurasi, per vowel, per speaker,
confusion matrix), `outputs/gaussian_predictions.csv` (prediksi dan NLL tiap file),
dan `outputs/gaussian_ablation.json`.

| Konfigurasi | Closed Recordings | Closed Speakers | Open Speakers | Rata-rata |
|---|---|---|---|---|
| full, tanpa CMVN (utama) | 100.00% | 93.33% | 68.75% | 87.36% |
| full, CMVN per speaker | 100.00% | 96.67% | 81.25% | 92.64% |
| diag, tanpa CMVN | 100.00% | 93.33% | 71.25% | 88.19% |
| diag, CMVN per speaker | 100.00% | 100.00% | 92.50% | 97.50% |

CMVN per speaker untuk data uji memakai statistik dari kelima rekaman vowel speaker
uji itu sendiri, jadi butuh semua ucapan speaker tersebut tersedia saat pengujian.

### DTW

Belum diimplementasikan.

### Pengujian dan akurasi

`src/evaluation.py` berisi `evaluate(records, predictions)` yang menghitung akurasi
total, per vowel, per speaker, dan confusion matrix. Fungsi ini bisa dipakai juga
untuk DTW. Closed recordings memakai audio template; closed speakers memakai
rekaman kedua dari speaker yang sama; open speakers memakai `load_open_set()`.
