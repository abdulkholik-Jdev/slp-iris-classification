# Single Layer Perceptron - Klasifikasi Biner Iris (Setosa vs Versicolor)

Implementasi Python yang mereplikasi spreadsheet SLP (Assignment 1).

- 4 fitur + bias, bobot awal 0,5, learning rate 0,1, 5 epoch
- Aktivasi sigmoid, loss = (g(z) - target)^2 (rata-rata per epoch), update bobot per sampel
- Latih: 40 Setosa + 40 Versicolor; validasi: 10 + 10 (bobot validasi = bobot akhir tiap epoch latih)

## Menjalankan
```
pip install -r requirements.txt
python slp_iris.py
```
Output (`output/`): `metrics.csv`, `accuracy_chart.png`, `loss_chart.png`.

## Hasil
| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|---|---|---|---|---|
| 1 | 0.449889 | 0.328951 | 0.5250 | 0.5000 |
| 2 | 0.037452 | 0.247289 | 0.9500 | 0.5000 |
| 3 | 0.024372 | 0.175892 | 0.9750 | 0.5000 |
| 4 | 0.017357 | 0.119381 | 0.9750 | 0.8500 |
| 5 | 0.012740 | 0.081581 | 0.9875 | 1.0000 |
