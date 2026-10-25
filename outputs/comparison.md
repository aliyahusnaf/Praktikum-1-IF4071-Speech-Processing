# Perbandingan DTW dan Model Gaussian

## Ringkasan akurasi

| Sistem | Closed Recordings | Closed Speakers | Open Speakers | Rata-rata |
|---|---|---|---|---|
| DTW | 96.67% | 96.67% | 82.50% | 91.94% |
| Gaussian | 100.00% | 93.33% | 68.75% | 87.36% |

## Tabel 1. Accuracy untuk setiap vowel

| Vowel | DTW Closed Recordings | DTW Closed Speakers | DTW Open Speakers | Gaussian Closed Recordings | Gaussian Closed Speakers | Gaussian Open Speakers |
|---|---|---|---|---|---|---|
| a | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 81.25% |
| i | 100.00% | 100.00% | 93.75% | 100.00% | 100.00% | 81.25% |
| u | 100.00% | 100.00% | 56.25% | 100.00% | 100.00% | 25.00% |
| e | 83.33% | 83.33% | 81.25% | 100.00% | 66.67% | 62.50% |
| o | 100.00% | 100.00% | 81.25% | 100.00% | 100.00% | 93.75% |
| Average Accuracy | 96.67% | 96.67% | 82.50% | 100.00% | 93.33% | 68.75% |

## Tabel 2. Closed experiments (per speaker)

| Speakers | DTW Closed Recordings | DTW Closed Speakers | Gaussian Closed Recordings | Gaussian Closed Speakers | Average |
|---|---|---|---|---|---|
| S1 | 100.00% | 100.00% | 100.00% | 80.00% | 95.00% |
| S2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| S3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| S4 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| S5 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| S6 | 80.00% | 80.00% | 100.00% | 80.00% | 85.00% |
| Average | 96.67% | 96.67% | 100.00% | 93.33% | 96.67% |

## Tabel 3. Open experiments (per speaker)

| Speakers | DTW Open Speakers | Gaussian Open Speakers | Average |
|---|---|---|---|
| G03S01 | 80.00% | 60.00% | 70.00% |
| G03S02 | 100.00% | 80.00% | 90.00% |
| G03S03 | 80.00% | 60.00% | 70.00% |
| G03S05 | 60.00% | 80.00% | 70.00% |
| G04S03 | 80.00% | 60.00% | 70.00% |
| G04S04 | 40.00% | 40.00% | 40.00% |
| G07S01 | 80.00% | 60.00% | 70.00% |
| G07S02 | 60.00% | 60.00% | 60.00% |
| G07S03 | 100.00% | 60.00% | 80.00% |
| G07S04 | 80.00% | 80.00% | 80.00% |
| G07S05 | 80.00% | 80.00% | 80.00% |
| G09S01 | 100.00% | 100.00% | 100.00% |
| G09S02 | 100.00% | 60.00% | 80.00% |
| G09S03 | 100.00% | 80.00% | 90.00% |
| G09S04 | 100.00% | 60.00% | 80.00% |
| G09S05 | 80.00% | 80.00% | 80.00% |
| Average | 82.50% | 68.75% | 75.62% |

## Confusion matrix

### DTW: Closed Recordings

| asli \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 6 | 0 | 0 | 0 | 0 |
| i | 0 | 6 | 0 | 0 | 0 |
| u | 0 | 0 | 6 | 0 | 0 |
| e | 0 | 1 | 0 | 5 | 0 |
| o | 0 | 0 | 0 | 0 | 6 |

### DTW: Closed Speakers

| asli \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 6 | 0 | 0 | 0 | 0 |
| i | 0 | 6 | 0 | 0 | 0 |
| u | 0 | 0 | 6 | 0 | 0 |
| e | 0 | 1 | 0 | 5 | 0 |
| o | 0 | 0 | 0 | 0 | 6 |

### DTW: Open Speakers

| asli \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 16 | 0 | 0 | 0 | 0 |
| i | 1 | 15 | 0 | 0 | 0 |
| u | 0 | 0 | 9 | 0 | 7 |
| e | 3 | 0 | 0 | 13 | 0 |
| o | 2 | 0 | 1 | 0 | 13 |

### Gaussian: Closed Recordings

| asli \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 6 | 0 | 0 | 0 | 0 |
| i | 0 | 6 | 0 | 0 | 0 |
| u | 0 | 0 | 6 | 0 | 0 |
| e | 0 | 0 | 0 | 6 | 0 |
| o | 0 | 0 | 0 | 0 | 6 |

### Gaussian: Closed Speakers

| asli \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 6 | 0 | 0 | 0 | 0 |
| i | 0 | 6 | 0 | 0 | 0 |
| u | 0 | 0 | 6 | 0 | 0 |
| e | 0 | 1 | 0 | 4 | 1 |
| o | 0 | 0 | 0 | 0 | 6 |

### Gaussian: Open Speakers

| asli \ prediksi | a | i | u | e | o |
|---|---|---|---|---|---|
| a | 13 | 0 | 0 | 0 | 3 |
| i | 0 | 13 | 1 | 2 | 0 |
| u | 0 | 0 | 4 | 0 | 12 |
| e | 0 | 2 | 1 | 10 | 3 |
| o | 1 | 0 | 0 | 0 | 15 |
