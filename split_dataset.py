"""
YOLO Dataset Splitter
─────────────────────
โครงสร้างที่ต้องมีก่อน:
    dataset/
    ├── images/   (รูปภาพ .jpg/.png)
    └── labels/   (YOLO .txt)

โครงสร้างที่จะได้หลัง split:
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
"""

import os, shutil, random, argparse, yaml
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset(
    dataset_dir: str,
    train_pct: float = 60.0,
    val_pct: float   = 40.0,
    class_names: list = None,
    seed: int = 42,
    copy: bool = True,         # True = copy, False = move
):
    dataset_dir = Path(dataset_dir).resolve()
    img_src = dataset_dir / "images"
    lbl_src = dataset_dir / "labels"

    assert img_src.exists(), f"ไม่พบโฟลเดอร์: {img_src}"
    assert lbl_src.exists(), f"ไม่พบโฟลเดอร์: {lbl_src}"

    # normalize %
    total = train_pct + val_pct
    train_r = train_pct / total
    val_r   = val_pct   / total

    # รวบรวมไฟล์รูปที่มี label คู่กัน
    pairs = []
    for f in sorted(img_src.iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        lbl_file = lbl_src / (f.stem + ".txt")
        if lbl_file.exists():
            pairs.append((f, lbl_file))
        else:
            print(f"  ⚠  ไม่พบ label สำหรับ: {f.name}  (ข้ามไป)")

    if not pairs:
        print("❌  ไม่มีไฟล์ที่จับคู่ได้ — ตรวจสอบโฟลเดอร์ images/ และ labels/")
        return

    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = max(1, round(n_total * train_r))
    n_val   = n_total - n_train

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:],
    }

    print(f"\n📊  ทั้งหมด {n_total} ไฟล์")
    print(f"   train  {n_train}  ({n_train/n_total*100:.1f}%)")
    print(f"   val    {n_val}   ({n_val/n_total*100:.1f}%)")
    print(f"   mode   {'copy' if copy else 'move'}\n")

    # สร้างโฟลเดอร์และคัดลอก/ย้ายไฟล์
    op = shutil.copy2 if copy else shutil.move

    for split, file_pairs in splits.items():
        out_img = img_src / split
        out_lbl = lbl_src / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img_f, lbl_f in file_pairs:
            op(str(img_f), str(out_img / img_f.name))
            op(str(lbl_f), str(out_lbl / lbl_f.name))

        print(f"  ✔  {split:5s}  →  {len(file_pairs)} ไฟล์")

    # สร้าง data.yaml
    if class_names is None:
        # พยายามอ่านจาก classes.txt ถ้ามี
        classes_txt = dataset_dir / "classes.txt"
        if classes_txt.exists():
            class_names = [l.strip() for l in classes_txt.read_text().splitlines() if l.strip()]
        else:
            class_names = ["object"]
            print("  ⚠  ไม่พบ classes.txt — ใช้ ['object'] แทน")

    yaml_data = {
        "path":  str(dataset_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_names),
        "names": class_names,
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

    print(f"\n✅  เสร็จแล้ว! data.yaml บันทึกที่:\n   {yaml_path}")
    print(f"\ndata.yaml:\n{yaml.dump(yaml_data, allow_unicode=True, sort_keys=False)}")


# ────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="แบ่ง YOLO dataset เป็น train/val",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=None,
        help="path ของโฟลเดอร์ dataset (ถ้าไม่ใส่จะถามทีหลัง)",
    )
    parser.add_argument(
        "--train", type=float, default=80.0,
        help="สัดส่วน train %% (default: 80)",
    )
    parser.add_argument(
        "--val", type=float, default=20.0,
        help="สัดส่วน val %% (default: 20)",
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="ชื่อ class เช่น --classes car person (ถ้าไม่ใส่จะอ่านจาก classes.txt)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--move", action="store_true",
        help="ย้ายไฟล์แทนการ copy (default: copy)",
    )

    args = parser.parse_args()

    # ถ้าไม่ได้ใส่ path ทาง CLI ให้ถามทาง input()
    if args.dataset_dir is None:
        print("=" * 50)
        print("  YOLO Dataset Splitter")
        print("=" * 50)
        dataset_dir = input("\nใส่ path โฟลเดอร์ dataset: ").strip().strip('"').strip("'")

        train_in = input("สัดส่วน train % [80]: ").strip()
        val_in   = input("สัดส่วน val   % [20]: ").strip()
        train_pct = float(train_in) if train_in else 80.0
        val_pct   = float(val_in)   if val_in   else 20.0

        mode_in = input("copy หรือ move? [copy]: ").strip().lower()
        copy    = mode_in != "move"
    else:
        dataset_dir = args.dataset_dir
        train_pct   = args.train
        val_pct     = args.val
        copy        = not args.move

    split_dataset(
        dataset_dir  = dataset_dir,
        train_pct    = train_pct,
        val_pct      = val_pct,
        class_names  = args.classes if args.dataset_dir else None,
        seed         = args.seed,
        copy         = copy,
    )