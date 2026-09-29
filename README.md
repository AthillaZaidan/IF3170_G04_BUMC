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
