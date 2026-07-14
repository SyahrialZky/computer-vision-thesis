"Project Directory"

project_root/
├── dataset_raw/                     # Data mentah sebelum di-split (Step 1 input)
│   ├── real/                        # Citra asli (kamera/Unsplash/Pexels)
│   └── fake/                        # Citra AI (GenImage, DiffusionDB, Bing/Leonardo)
│
├── dataset/                         # Output Step 1: hasil split + resize 224x224
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
├── src/                             # Semua kode modular
│   ├── step1_preprocessing.py       # ✅ sudah dibuat
│   ├── step2_frequency_transform.py # ✅ sudah dibuat (rgb_to_fft, rgb_to_dct)
│   ├── step3_dataset.py             # AIDetectionDataset(mode="rgb"/"fft"/"dct"/"fusion")
│   ├── step4_train.py               # training loop, early stopping, save best model
│   ├── step5_evaluate.py            # metrik test set, confusion matrix, ROC
│   ├── step6_gradcam.py             # Grad-CAM per skenario
│   ├── model.py                     # builder EfficientNet-B0 + custom classifier head
│   └── utils.py                     # set_seed(42), config loader, helper bersama
│
├── notebooks/                       # Eksplorasi/debug interaktif (opsional, Colab-friendly)
│   └── exploration.ipynb
│
├── checkpoints/                     # Output Step 4: model terbaik per skenario
│   ├── S1_rgb_best.pth
│   ├── S2_fft_best.pth
│   ├── S3_dct_best.pth
│   └── S4_fusion_best.pth
│
├── outputs/                         # Semua artefak untuk Bab 4 skripsi
│   ├── figures/
│   │   ├── fft_dct_samples.png          # dari Step 2
│   │   ├── training_curves_S1.png … S4.png
│   │   ├── confusion_matrix_S1.png … S4.png
│   │   ├── roc_curve_comparison.png     # semua skenario, 1 plot
│   │   └── gradcam/
│   │       ├── S1_rgb_sample01.png
│   │       └── S2_fft_sample01.png …
│   ├── logs/
│   │   └── training_log_S1.csv … S4.csv  # loss/acc per epoch
│   └── reports/
│       └── classification_report_S1.txt … S4.txt  # sklearn report
│
├── requirements.txt
└── README.md