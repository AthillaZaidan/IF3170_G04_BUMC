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

Bonus yang dikerjakan: seluruh varian Hill-Climbing (steepest-ascent, stochastic, sideways move,
random-restart), objective function alternatif berbasis ETA (`value / eta`), dan multi-truk.

## Struktur Repository

```
src/
  core/          model, representasi state, constraint, objective, repair, neighbor, batas atas
  algorithms/    hill_climbing.py, simulated_annealing.py, genetic.py
  experiments/   hc_experiment.py, sa_experiment.py, ga_experiment.py
  viz/           visualisasi 3D state dan plot hasil
  main.py        CLI untuk satu kali run
  generate_instance.py
data/instances/  instance small, medium, large (seed 42)
output/          hasil run: output/<algoritma>/<instance>/<konfigurasi>/seed<N>/
docs/            laporan (main.pdf) beserta sumber LaTeX
tests/           unit test (pytest)
```

## Setup

Membutuhkan Python 3.10 atau lebih baru.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Menjalankan unit test:

```bash
python -m pytest -q
```

## Cara Menjalankan

Semua perintah dijalankan dari root repository.

### Satu kali run

Setiap run menyimpan `result.json` (parameter, seed, durasi, state awal dan akhir, riwayat nilai),
`initial.png` dan `final.png` (visualisasi 3D state), serta `curve.png` (objective terhadap iterasi)
ke `output/<algoritma>/<instance>/single/`. Opsi `--objective eta` memakai objective ETA, dan `--out`
mengganti folder keluaran.

```bash
# Hill-Climbing (varian: steepest, stochastic, sideways, random-restart)
python -m src.main hc --instance data/instances/medium.json --variant steepest --iters 100 --seed 1
python -m src.main hc --instance data/instances/medium.json --variant sideways --max-sideways 20 --seed 1
python -m src.main hc --instance data/instances/medium.json --variant random-restart --max-restarts 5 --seed 1

# Simulated Annealing (ditambah eet.png: e^(dE/T) per iterasi)
python -m src.main sa --instance data/instances/medium.json --iters 20000 --p0 0.01 --seed 1

# Genetic Algorithm (curve.png berisi objective maksimum dan rata-rata populasi)
python -m src.main ga --instance data/instances/medium.json --pop 50 --gens 100 --seed 1

# Bonus: objective ETA
python -m src.main sa --instance data/instances/medium.json --objective eta --seed 1
```

### Multi-truk

Opsi `--trucks N` tersedia pada HC (semua varian), SA, GA, generator instance, dan skrip
eksperimen. Armada terdiri dari N truk identik, masing-masing memakai dimensi dan kapasitas
`truck` dari instance. Nilai objective adalah total paket yang diangkut seluruh armada;
setiap paket hanya bisa berada di satu truk. Paket dapat dipindahkan atau ditukar antartuk.
Constraint berat, dinding, overlap, penopang, dan fragile diperiksa per truk.

```bash
python -m src.main hc --instance data/instances/small.json --trucks 2 --variant steepest
python -m src.main sa --instance data/instances/small.json --trucks 2 --iters 2000
python -m src.main ga --instance data/instances/small.json --trucks 2 --pop 50 --gens 100
python -m src.experiments.hc_experiment --instance data/instances/small.json --trucks 2
python -m src.generate_instance --size small --trucks 2 --seed 42
```

Jumlah truk juga dapat disimpan sebagai `"truck_count": 2` pada level teratas JSON instance.
Jika `--trucks` diberikan, nilainya menggantikan `truck_count` dari file. Tanpa keduanya,
program memakai satu truk. File JSON lama tetap dapat dibaca.

Output multi-truk memakai folder instance bersufiks `_trucksN`, misalnya
`output/hc/small_trucks2/single/`. `initial.png` dan `final.png` menampilkan setiap truk,
termasuk truk kosong, beserta berat muatannya. `result.json` menyimpan konfigurasi armada
di `problem` dan `truck_id` berbasis nol pada setiap placement; abaikan `truck_id` jika
`in_truck` bernilai `false`.

Batas atas eksperimen memakai kapasitas berat dan volume gabungan armada, sambil
mengecualikan paket yang tidak muat dalam satu truk. Nilainya merupakan relaksasi
knapsack, sehingga pembagian paket antartuk masih dapat membatasi solusi sebenarnya.

### Eksperimen

Setiap skrip eksperimen menjalankan beberapa seed per konfigurasi dan menyimpan `summary.csv` serta
plot pembanding di `output/<algoritma>/<instance>/`. Ganti `medium` dengan `small` atau `large` untuk
instance lain.

```bash
# HC: keempat varian, 3 seed (instance large memakai --max-restarts 2)
python -m src.experiments.hc_experiment --instance data/instances/medium.json

# SA: grid parameter, lihat bagian Heuristik Simulated Annealing di bawah
python -m src.experiments.sa_experiment --instance data/instances/medium.json --iterations 20000 --p0 0.01 --seeds 1 2 3

# GA: populasi tetap 50 dengan generasi 50/100/200, generasi tetap 100 dengan populasi 20/50/100, 3 seed
python -m src.experiments.ga_experiment --instance data/instances/medium.json
```

Membangkitkan ulang instance:

```bash
python -m src.generate_instance --size medium --seed 42
```

### Laporan

```bash
cd docs && latexmk -pdf main.tex
```

## Heuristik Simulated Annealing

Implementasi: `src/algorithms/simulated_annealing.py`. Setiap iterasi mengambil satu tetangga acak; jika
`dE > 0` tetangga diterima, jika tidak diterima dengan peluang `e^(dE/T)`.

```bash
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

## Pembagian Tugas

| Nama | NIM | Tugas |
|---|---|---|
| Athilla Zaidan Zidna Fann | 13524068 | Genetic Algorithm |
| Jason Edward Salim | 13524034 | Simulated Annealing |
| Raynard Fausta | 13524052 | Hill-Climbing, multi-truk (bonus) |
