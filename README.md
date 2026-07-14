```markdown
# AI Image Detection: Real vs Fake (AI-Generated) Image Classification

Repositori ini berisi implementasi proyek penelitian klasifikasi citra untuk mendeteksi apakah sebuah gambar merupakan citra asli (*real*) atau citra hasil generator AI (*fake/generated*). Pendekatan yang digunakan mencakup analisis domain spasial (RGB), domain frekuensi (FFT & DCT), serta kombinasi keduanya (*fusion*) menggunakan arsitektur **EfficientNet-B0**.

Proyek ini dirancang secara modular untuk mempermudah eksperimen, evaluasi, hingga visualisasi interpretabilitas model menggunakan Grad-CAM untuk kebutuhan Bab 4 Skripsi.

---

## 📂 Struktur Direktori Proyek

```text
project_root/
├── dataset_raw/                     # Data mentah sebelum di-split (Step 1 Input)
│   ├── real/                        # Citra asli (Kamera, Unsplash, Pexels, dll.)
│   └── fake/                        # Citra AI (GenImage, DiffusionDB, Bing/Leonardo, dll.)
│
├── dataset/                         # Output Step 1: Hasil split & resize 224x224
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
│
├── src/                             # Sumber kode utama (Modular Scripts)
│   ├── step1_preprocessing.py       # Pemisahan dataset (Train/Val/Test) & Resize citra
│   ├── step2_frequency_transform.py # Transformasi domain frekuensi (RGB to FFT & DCT)
│   ├── step3_dataset.py             # Custom PyTorch Dataset (Mode: RGB, FFT, DCT, Fusion)
│   ├── step4_train.py               # Training loop, Early Stopping, & Save Best Model
│   ├── step5_evaluate.py            # Kalkulasi metrik test set, Confusion Matrix, & Kurva ROC
│   ├── step6_gradcam.py             # Visualisasi Grad-CAM per skenario eksperimen
│   ├── model.py                     # Builder arsitektur EfficientNet-B0 + Custom Classifier Head
│   └── utils.py                     # Fungsi pembantu (set_seed, config loader, helper bersama)
│
├── notebooks/                       # Eksplorasi & debugging interaktif
│   └── exploration.ipynb            # Notebook analisis awal (Colab-friendly)
│
├── checkpoints/                     # Output Step 4: Model terbaik per skenario
│   ├── S1_rgb_best.pth              # Bobot model terbaik skenario 1 (RGB)
│   ├── S2_fft_best.pth              # Bobot model terbaik skenario 2 (FFT)
│   ├── S3_dct_best.pth              # Bobot model terbaik skenario 3 (DCT)
│   └── S4_fusion_best.pth           # Bobot model terbaik skenario 4 (Fusion)
│
├── outputs/                         # Semua artefak hasil untuk kebutuhan Bab 4 Skripsi
│   ├── figures/                     # Visualisasi grafik dan citra hasil proses
│   │   ├── fft_dct_samples.png      # Sampel hasil transformasi frekuensi dari Step 2
│   │   ├── training_curves_S1.png   # Kurva Loss & Akurasi (S1 s.d S4)
│   │   ├── confusion_matrix_S1.png  # Confusion Matrix pengujian (S1 s.d S4)
│   │   ├── roc_curve_comparison.png # Perbandingan kurva ROC seluruh skenario (1 plot)
│   │   └── gradcam/                 # Heatmap hasil representasi visual model
│   │       ├── S1_rgb_sample01.png
│   │       └── S2_fft_sample01.png
│   ├── logs/                        # Catatan performa per epoch
│   │   └── training_log_S1.csv      # File CSV log training (S1 s.d S4)
│   └── reports/                     # Laporan metrik klasifikasi tekstual
│       └── classification_report_S1.txt # Precision, Recall, F1-Score (S1 s.d S4)
│
├── requirements.txt                 # Daftar dependensi library Python
└── README.md                        # Dokumentasi proyek