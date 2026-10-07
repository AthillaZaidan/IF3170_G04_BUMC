# IF3170_G04_BUMC

Tugas Besar 1 IF3170 Inteligensi Artifisial 2026/2027.

## Deskripsi Tugas

Program ini mencari konfigurasi pengisian paket paling optimal pada sebuah peti
kemas truk dengan algoritma *local search*. Konfigurasi dianggap optimal jika
**total value paket di dalam peti kemas maksimal**.

Setiap paket berbentuk balok dan memiliki dimensi, orientasi, value, berat,
status pecah-belah (*fragile*), dan ETA. Truk memiliki satu peti kemas dengan
dimensi dan kapasitas berat maksimum. Tidak semua paket harus dimasukkan; paket
boleh berada sepenuhnya di luar peti kemas.

Konfigurasi harus memenuhi constraint berikut:

- Paket tidak boleh melayang. Minimal satu titik alasnya harus bertumpu pada
  lantai peti kemas atau pada bagian atas paket lain.
- Paket tidak boleh menembus dinding peti kemas.
- Barang pecah-belah tidak boleh menjadi penopang paket lain.
- Total berat paket di dalam peti kemas tidak boleh melebihi kapasitas truk.

State awal dibangkitkan secara acak. Langkah yang diperbolehkan untuk berpindah ke
state tetangga adalah menukar dua paket, memindahkan sebuah paket ke koordinat
lain, dan merotasi paket 90° terhadap sumbu x, y, atau z.

Algoritma yang diimplementasikan:

1. Hill-Climbing
2. Simulated Annealing
3. Genetic Algorithm

## Simulated Annealing

Implementasi: `src/algorithms/simulated_annealing.py`. Setiap iterasi mengambil satu tetangga acak; jika
`dE > 0` tetangga diterima, jika tidak diterima dengan peluang `e^(dE/T)`.

```bash
# satu kali run (state awal/akhir, kurva objective, kurva e^(dE/T), durasi, frekuensi stuck)
python -m src.main sa --instance data/instances/medium.json --iters 20000 --p0 0.01 --seed 1

# satu kali run GA
python -m src.main ga --instance data/instances/medium.json --pop 50 --gens 100 --seed 1

# eksperimen SA: grid schedule x p0 x t_min x iterasi x ratio_bias, 5 seed per konfigurasi
python -m src.experiments.sa_experiment --instance data/instances/medium.json \
    --iterations 20000 --p0 0.01 --ratio-bias 0 0.5
```

`sa_experiment` memberi setiap run seed sendiri (dari nama konfigurasi dan nomor seed), sehingga
konfigurasi yang berbeda adalah run yang saling bebas, sama seperti eksperimen GA. Opsi lain:
`--schedules geometric linear`, `--t-min`, `--patience`, `--seeds`, dan `--out` (default `output/sa`).

Setiap run menyimpan `result.json`, `initial.png`, `final.png`, `curve.png` (objective dan suhu per
iterasi) dan `eet.png` (`e^(dE/T)` per iterasi). Eksperimen juga menyimpan `summary.csv` (termasuk
`gap_to_bound` dan `stuck_count`), `compare_best.png`, `final_value_box.png`, dan `stuck_count_box.png`.

Heuristik yang dipakai:

- **Repair setiap tetangga:** tetangga acak dari `random_neighbor` langsung di-`repair`, sehingga SA
  hanya berjalan di state valid dan nilai objective = total value (sama seperti HC dan GA).
- **Kalibrasi suhu awal:** jika `--t0` tidak diberikan, `T0 = -rata2(|dE| langkah memburuk) / ln(p0)`
  dari 100 tetangga acak state awal, sehingga langkah memburuk rata-rata diterima dengan peluang `p0`
  (Ben-Ameur, *Comput. Optim. Appl.* 29, 2004).
- **Jadwal suhu dengan titik ujung tetap:** suhu turun dari `T0` ke `t_min` tepat dalam `--iters`
  iterasi, secara geometrik (`T <- alpha * T`) atau linear, sehingga banyak iterasi adalah satu-satunya
  parameter biaya.
- **Simpan state terbaik:** hasil akhir adalah state terbaik yang pernah dikunjungi.
- **Ratio swap (opsional, `--ratio-bias`):** dengan peluang `ratio_bias`, tetangga yang diusulkan adalah swap
  antara paket di dalam truk dengan value/berat rendah dan paket di luar dengan value/berat tinggi. Langkahnya
  tetap swap biasa, hanya pemilihan paketnya yang dibias. Pada 3 seed, `--ratio-bias 0.5` menaikkan
  rata-rata value 1.6% di `large` dan 0.8% di `medium` (lihat `output/sa/<instance>/summary.csv`).
- **Definisi stuck:** SA dianggap *stuck* di local optimum jika nilai objective state saat ini tidak
  berubah selama `--patience` (default 500) iterasi berturut-turut; satu plateau dihitung sekali.

`src/core/bounds.py` menghitung batas atas global optimum: knapsack 0/1 dengan kapasitas berat dan
volume, diselesaikan tepat dengan DP. Karena geometri diabaikan, nilainya tidak lebih kecil dari global
optimum, sehingga `1 - value / batas_atas` adalah batas atas jarak hasil ke global optimum.
