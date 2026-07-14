"""
Step 2: Transformasi Domain Frekuensi (FFT & DCT)
====================================================
Skripsi: "Analisis Komparatif Ekstraksi Fitur Domain Frekuensi (FFT dan DCT)
pada Arsitektur CNN Ringan untuk Deteksi Citra Buatan Generative AI"

Berisi:
    - rgb_to_fft(image)  -> magnitude spectrum FFT, 3 channel, uint8, 224x224
    - rgb_to_dct(image)  -> log-magnitude DCT, 3 channel, uint8, 224x224
    - visualize_samples()-> menampilkan gambar asli berdampingan dengan
                            hasil FFT dan DCT-nya, disimpan sebagai .png

Cara menjalankan (mode standalone untuk uji coba & visualisasi):
    python step2_frequency_transform.py \
        --sample_dir dataset/train/real \
        --n_samples 5 \
        --img_size 224 \
        --output_path outputs/fft_dct_samples.png

Fungsi rgb_to_fft() dan rgb_to_dct() didesain untuk diimpor langsung oleh
Custom Dataset Class (Step 3) sehingga transformasi konsisten di seluruh
pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
from scipy.fft import dctn
import matplotlib.pyplot as plt

DEFAULT_IMG_SIZE = 224
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Menormalisasi array float ke rentang 0-255 dan mengonversinya ke uint8.

    Menggunakan min-max normalization per-gambar (bukan per-dataset) karena
    setiap transformasi frekuensi punya rentang nilai yang sangat bervariasi
    antar gambar.
    """
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        # Hindari pembagian dengan nol pada gambar yang nyaris konstan
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return (normalized * 255).astype(np.uint8)


def rgb_to_fft(image: np.ndarray, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Mengonversi citra RGB menjadi magnitude spectrum FFT 2D.

    Pipeline:
        1. RGB -> Grayscale
        2. 2D FFT (numpy.fft.fft2)
        3. fftshift agar frekuensi rendah berada di tengah
        4. Magnitude spectrum: 20 * log(|F| + 1)
        5. Normalisasi ke 0-255 (uint8)
        6. Duplikasi ke 3 channel agar kompatibel dengan EfficientNet
        7. Resize ke (img_size, img_size)

    Args:
        image: array RGB, shape (H, W, 3), dtype uint8 atau float.
        img_size: ukuran output akhir (persegi).

    Returns:
        Array uint8, shape (img_size, img_size, 3), representasi FFT magnitude.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Input harus citra RGB (H, W, 3), didapat shape {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    fft_result = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft_result)

    magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1)
    fft_uint8 = _normalize_to_uint8(magnitude_spectrum)

    fft_rgb = cv2.cvtColor(fft_uint8, cv2.COLOR_GRAY2RGB)
    fft_resized = cv2.resize(fft_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    return fft_resized


def rgb_to_dct(image: np.ndarray, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """
    Mengonversi citra RGB menjadi representasi log-magnitude DCT 2D.

    Pipeline:
        1. RGB -> Grayscale
        2. 2D DCT (scipy.fft.dctn, tipe II, norm='ortho')
        3. Ambil nilai absolut
        4. Transformasi logaritmik: log(|DCT| + 1)
        5. Normalisasi ke 0-255 (uint8)
        6. Duplikasi ke 3 channel
        7. Resize ke (img_size, img_size)

    Args:
        image: array RGB, shape (H, W, 3), dtype uint8 atau float.
        img_size: ukuran output akhir (persegi).

    Returns:
        Array uint8, shape (img_size, img_size, 3), representasi DCT log-magnitude.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Input harus citra RGB (H, W, 3), didapat shape {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    dct_result = dctn(gray, type=2, norm="ortho")
    dct_abs = np.abs(dct_result)
    dct_log = np.log(dct_abs + 1)

    dct_uint8 = _normalize_to_uint8(dct_log)
    dct_rgb = cv2.cvtColor(dct_uint8, cv2.COLOR_GRAY2RGB)
    dct_resized = cv2.resize(dct_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    return dct_resized


def load_rgb_image(path: Path, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Membaca gambar dari disk sebagai array RGB uint8, sudah di-resize."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise IOError(f"Gagal membaca gambar: {path}")
    bgr = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def collect_sample_paths(sample_dir: Path, n_samples: int) -> List[Path]:
    """Mengambil n_samples path gambar pertama dari sebuah folder."""
    paths = sorted(
        p for p in sample_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS
    )
    if len(paths) == 0:
        raise ValueError(f"Tidak ada gambar ditemukan di {sample_dir}")
    return paths[:n_samples]


def visualize_samples(
    sample_paths: List[Path],
    img_size: int,
    output_path: Path,
) -> None:
    """
    Menampilkan grid: setiap baris = 1 sampel gambar, kolom = [RGB, FFT, DCT].

    Hasil visualisasi disimpan sebagai file .png ke output_path, sesuai
    kebutuhan Bab 4 skripsi (sample visualisasi FFT & DCT).
    """
    n = len(sample_paths)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)  # pastikan tetap 2D saat hanya 1 sampel

    col_titles = ["RGB (Asli)", "FFT Magnitude Spectrum", "DCT Log-Magnitude"]

    for row, img_path in enumerate(sample_paths):
        rgb_img = load_rgb_image(img_path, img_size)
        fft_img = rgb_to_fft(rgb_img, img_size)
        dct_img = rgb_to_dct(rgb_img, img_size)

        for col, img in enumerate([rgb_img, fft_img, dct_img]):
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)

        axes[row, 0].set_ylabel(img_path.name, fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Step 2] Visualisasi sampel disimpan di: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Uji coba & visualisasi transformasi FFT/DCT pada sampel gambar."
    )
    parser.add_argument("--sample_dir", type=str, required=True,
                         help="Folder berisi gambar sampel (misal dataset/train/real)")
    parser.add_argument("--n_samples", type=int, default=5,
                         help="Jumlah sampel yang divisualisasikan (5-10 disarankan)")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--output_path", type=str, default="outputs/fft_dct_samples.png",
                         help="Path file .png untuk menyimpan hasil visualisasi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sample_dir = Path(args.sample_dir)
    output_path = Path(args.output_path)

    print(f"[Step 2] Mengambil {args.n_samples} sampel dari: {sample_dir}")
    sample_paths = collect_sample_paths(sample_dir, args.n_samples)
    for p in sample_paths:
        print(f"  - {p.name}")

    print("\n[Step 2] Menjalankan transformasi FFT & DCT pada sampel...")
    visualize_samples(sample_paths, args.img_size, output_path)

    print("[Step 2] Selesai.")


if __name__ == "__main__":
    main()
