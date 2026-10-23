# Pengenalan Vowel Bahasa Indonesia

Praktikum IF4071: pengenalan vowel a, i, u, e, o dengan DTW dan Model Gaussian.

## Dataset

6 speaker, masing-masing 5 vowel dengan 2 rekaman: total 60 audio.
Format audio: WAV PCM 16-bit, mono, 16000 Hz.

- `data/raw/`: rekaman asli.
- `data/processed/template/`: 30 audio rekaman pertama.
- `data/processed/test/closed/`: 30 audio rekaman kedua.
- `data/processed/test/open/`: belum diisi.

Nama file: `S1_a1.wav` = speaker S1, vowel a, rekaman 1.
Speaker: S1 Aliya, S2 Farrel, S3 Fithra, S4 Kefas, S5 Rakdaf, S6 Timur.

## Menjalankan program

Jalankan dari folder proyek dengan Python 3.9+:

```powershell
python -m pip install -r requirements.txt
python -m src.preprocessing
python -m unittest discover -s tests -v
```

### Preprocessing dan data loading

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

Belum diimplementasikan. Input dari loader: array audio dan sample rate.
Output yang dibutuhkan: fitur 39 dimensi per frame.

### DTW dan Model Gaussian

Belum diimplementasikan.

### Pengujian dan akurasi

Belum diimplementasikan. Closed recordings memakai audio template;
closed speakers memakai rekaman kedua dari speaker yang sama.
