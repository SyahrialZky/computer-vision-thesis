"""
Step 1: Preprocessing & Split Data
====================================
Skripsi: "Analisis Komparatif Ekstraksi Fitur Domain Frekuensi (FFT dan DCT)
pada Arsitektur CNN Ringan untuk Deteksi Citra Buatan Generative AI"

Tugas script ini:
1. Membaca folder dataset mentah (raw), diasumsikan berstruktur:
       <raw_dir>/real/*.jpg|png
       <raw_dir>/fake/*.jpg|png
2. Melakukan stratified split 70:15:15 (train/val/test) agar proporsi
   kelas real vs fake tetap seimbang di setiap split.
3. Melakukan resize seluruh gambar ke 224x224 (sesuai input EfficientNet-B0).
4. Menyimpan hasil ke struktur folder final:
       <output_dir>/train/real, /train/fake
       <output_dir>/val/real,   /val/fake
       <output_dir>/test/real,  /test/fake

Cara menjalankan:
    python step1_preprocessing.py \
        --raw_dir dataset_raw \
        --output_dir dataset \
        --img_size 224 \
        --train_ratio 0.70 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --seed 42

Catatan:
- Kelas diharapkan sudah relatif balanced di sumber data. Jika tidak,
  gunakan --balance_classes untuk melakukan undersampling otomatis
  terhadap kelas mayoritas sehingga rasio real:fake menjadi 1:1.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------------------------------------------------
# Konfigurasi default (bisa dioverride lewat argparse)
# -----------------------------------------------------------------------
DEFAULT_IMG_SIZE = 224
DEFAULT_SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["real", "fake"]  # label 0 = real, 1 = fake


def set_seed(seed: int) -> None:
    """Set random seed untuk reproducibility."""
    random.seed(seed)


def list_images_by_class(raw_dir: Path) -> Dict[str, List[Path]]:
    """
    Membaca folder dataset mentah dan mengelompokkan path gambar per kelas.

    Args:
        raw_dir: path ke folder dataset mentah, berisi subfolder 'real' dan 'fake'.

    Returns:
        Dict berisi {"real": [path, ...], "fake": [path, ...]}
    """
    class_to_paths: Dict[str, List[Path]] = {}

    for cls in CLASS_NAMES:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"Folder kelas '{cls}' tidak ditemukan di {raw_dir}. "
                f"Pastikan struktur dataset mentah adalah <raw_dir>/real dan <raw_dir>/fake."
            )
        paths = sorted(
            p for p in cls_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS
        )
        if len(paths) == 0:
            raise ValueError(f"Tidak ada file gambar valid ditemukan di {cls_dir}")
        class_to_paths[cls] = paths

    return class_to_paths


def balance_classes(
    class_to_paths: Dict[str, List[Path]], seed: int
) -> Dict[str, List[Path]]:
    """
    Melakukan undersampling pada kelas mayoritas agar rasio real:fake = 1:1.

    Args:
        class_to_paths: dict {kelas: list path}
        seed: random seed untuk sampling.

    Returns:
        Dict baru dengan jumlah sampel per kelas yang sudah seimbang.
    """
    rng = random.Random(seed)
    min_count = min(len(paths) for paths in class_to_paths.values())

    balanced: Dict[str, List[Path]] = {}
    for cls, paths in class_to_paths.items():
        if len(paths) > min_count:
            balanced[cls] = rng.sample(paths, min_count)
        else:
            balanced[cls] = paths

    print(f"[Balance] Jumlah sampel per kelas setelah balancing: {min_count}")
    return balanced


def stratified_split(
    class_to_paths: Dict[str, List[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Melakukan stratified split per kelas menjadi train/val/test.

    Stratifikasi dilakukan secara manual per kelas (bukan lewat argumen
    `stratify` sklearn) agar proporsi kelas persis sama di setiap split,
    karena kita men-split masing-masing kelas secara terpisah lalu
    menggabungkannya kembali.

    Args:
        class_to_paths: dict {kelas: list path gambar}.
        train_ratio, val_ratio, test_ratio: proporsi split, harus berjumlah 1.0.
        seed: random seed.

    Returns:
        Dict {"train": {kelas: [path,...]}, "val": {...}, "test": {...}}
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not abs(ratio_sum - 1.0) < 1e-6:
        raise ValueError(f"train+val+test ratio harus = 1.0, saat ini = {ratio_sum}")

    splits: Dict[str, Dict[str, List[Path]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    for cls, paths in class_to_paths.items():
        # Split pertama: train vs (val+test)
        train_paths, temp_paths = train_test_split(
            paths,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
        )
        # Split kedua: bagi temp menjadi val dan test sesuai proporsi relatif
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths = train_test_split(
            temp_paths,
            train_size=relative_val_ratio,
            random_state=seed,
            shuffle=True,
        )

        splits["train"][cls] = train_paths
        splits["val"][cls] = val_paths
        splits["test"][cls] = test_paths

        print(
            f"[Split] Kelas '{cls}': train={len(train_paths)}, "
            f"val={len(val_paths)}, test={len(test_paths)}"
        )

    return splits


def resize_and_save(src_path: Path, dst_path: Path, img_size: int) -> None:
    """
    Membuka gambar, resize ke (img_size, img_size), konversi ke RGB,
    lalu menyimpannya ke lokasi tujuan.

    Menggunakan try/except agar satu file korup tidak menghentikan seluruh
    proses preprocessing; file yang gagal dibaca akan dilewati dan dicatat.
    """
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize((img_size, img_size), Image.BICUBIC)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path)
    except Exception as e:  # noqa: BLE001 - sengaja tangkap semua error file I/O
        print(f"[WARNING] Gagal memproses {src_path}: {e}")


def build_output_structure(
    splits: Dict[str, Dict[str, List[Path]]],
    output_dir: Path,
    img_size: int,
) -> None:
    """
    Melakukan resize dan menyalin seluruh gambar hasil split ke struktur
    folder output final: <output_dir>/<split>/<class>/<filename>
    """
    for split_name, class_to_paths in splits.items():
        for cls, paths in class_to_paths.items():
            desc = f"Processing {split_name}/{cls}"
            for src_path in tqdm(paths, desc=desc):
                dst_path = output_dir / split_name / cls / src_path.name
                resize_and_save(src_path, dst_path, img_size)


def print_summary(output_dir: Path) -> None:
    """Mencetak ringkasan jumlah file per split/kelas sebagai sanity check."""
    print("\n=== Ringkasan Dataset Hasil Split ===")
    for split_name in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            folder = output_dir / split_name / cls
            count = len(list(folder.glob("*"))) if folder.exists() else 0
            print(f"{split_name:5s} / {cls:5s}: {count} gambar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: Preprocessing & stratified split dataset real vs fake."
    )
    parser.add_argument("--raw_dir", type=str, required=True,
                         help="Path folder dataset mentah, berisi subfolder real/ dan fake/")
    parser.add_argument("--output_dir", type=str, default="dataset",
                         help="Path folder output hasil split (default: dataset)")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE,
                         help="Ukuran resize gambar, default 224")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--balance_classes", action="store_true",
                         help="Undersample kelas mayoritas agar rasio real:fake = 1:1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    print(f"[Step 1] Membaca dataset mentah dari: {raw_dir}")
    class_to_paths = list_images_by_class(raw_dir)
    for cls, paths in class_to_paths.items():
        print(f"  Kelas '{cls}': {len(paths)} gambar ditemukan")

    if args.balance_classes:
        class_to_paths = balance_classes(class_to_paths, seed=args.seed)

    print("\n[Step 1] Melakukan stratified split (train/val/test)...")
    splits = stratified_split(
        class_to_paths,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"\n[Step 1] Resize gambar ke {args.img_size}x{args.img_size} "
          f"dan menyimpan ke: {output_dir}")
    build_output_structure(splits, output_dir, args.img_size)

    print_summary(output_dir)
    print("\n[Step 1] Selesai. Dataset siap digunakan untuk training.")


if __name__ == "__main__":
    main()
