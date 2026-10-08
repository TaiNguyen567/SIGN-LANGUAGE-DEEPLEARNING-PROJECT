"""Filter complex dataset classes and generate lightweight basic daily communication datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# PRESET 1: Core Daily Communication (47 classes)
# Essential daily needs: Greetings, basic eating/drinking, personal hygiene, immediate family, core questions.
CORE_DAILY_CLASSES = [
    # 1. Chào hỏi & Xã giao (5)
    "chào",
    "tạm biệt",
    "xin lỗi",
    "không có chi",
    "chúc mừng",
    # 2. Nhu cầu ăn uống thiết yếu (11)
    "ăn uống",
    "cơm",
    "bữa cơm",
    "đói không？",
    "no",
    "khát nước",
    "chai nước",
    "bánh mì",
    "phở",
    "sữa",
    "trà",
    # 3. Sinh hoạt, Vệ sinh & Sức khỏe (10)
    "đi vệ sinh",
    "nhà vệ sinh",
    "rửa tay",
    "rửa mặt",
    "đánh răng",
    "mặc quần áo",
    "giấc ngủ",
    "sức khỏe",
    "mệt không？",
    "đau bụng",
    # 4. Xưng hô & Người thân (8)
    "bạn",
    "các bạn",
    "chúng tôi (đại từ)",
    "anh em",
    "anh chị",
    "em trai",
    "em gái",
    "bố mẹ",
    # 5. Giao tiếp, Nhu cầu & Hỏi đáp (13)
    "tên là gì？",
    "bao nhiêu？",
    "mấy giờ？",
    "có … không？",
    "đúng không？",
    "muốn",
    "muốn không？",
    "cần không？",
    "không cần",
    "hiểu",
    "không hiểu",
    "giúp đỡ",
    "về 5 (về nhà)",
]

# PRESET 2: Standard Daily Communication (116 classes)
# Adds common dishes, utensils, extended family, daily questions/emotions, numbers 0-10, and daily times.
STANDARD_DAILY_CLASSES = CORE_DAILY_CLASSES + [
    # Ăn uống mở rộng & Đồ dùng ăn uống (19)
    "đồ ăn",
    "đồ uống",
    "cơm hộp",
    "cơm rang",
    "nước ép trái cây",
    "bún chả",
    "bún đậu",
    "bánh cuốn",
    "rau",
    "cá kho",
    "chả cá",
    "trứng",
    "sữa bò",
    "trà nóng",
    "pha cà phê",
    "ngon miệng",
    "không ngon",
    "bát",
    "đĩa",
    "đũa",
    "thìa",
    "cốc",
    # Sinh hoạt & Sức khỏe mở rộng (6)
    "toa lét",
    "vệ sinh cá nhân",
    "cuộn giấy vệ sinh",
    "bàn chải đánh răng",
    "mệt mỏi",
    "sốt",
    # Xưng hô & Gia đình mở rộng (8)
    "anh chị em",
    "bố",
    "cha mẹ",
    "ông bà",
    "con trai",
    "con gái",
    "trẻ con／con nít",
    "người yêu",
    # Giao tiếp & Cảm xúc thường nhật (16)
    "là gì？",
    "bao giờ？",
    "thế nào？",
    "vì sao？",
    "còn bạn？",
    "phải không？",
    "ai",
    "không muốn",
    "nói chuyện",
    "lắng nghe",
    "hỏi",
    "đi",
    "vui mừng",
    "buồn thảm",
    "lo sợ",
    "sợ không？",
    # Thời gian & Số đếm cơ bản (16)
    "buổi sáng",
    "buổi trưa",
    "buổi chiều",
    "buổi tối",
    "ban ngày",
    "ban đêm",
    "0 (số không)",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
]

# PRESET 3: Extended Daily Life (169 classes)
# Full daily lifestyle dataset without any academic, geographical, formulaic, or rare esoteric words.
EXTENDED_DAILY_CLASSES = STANDARD_DAILY_CLASSES + [
    "cơm bình dân",
    "bún mắm",
    "sữa hộp",
    "trà sữa",
    "trà đá",
    "đun nấu",
    "nấu nướng",
    "mặc",
    "quần đùi",
    "quần bò",
    "áo",
    "mơ ngủ",
    "nghỉ ngơi",
    "đau chân",
    "đau tay",
    "đau mắt",
    "ốm đau",
    "hiệu thuốc",
    "thuốc bổ",
    "anh hai, anh cả",
    "bế em",
    "má (giống： mẹ)",
    "ông nội",
    "ông ngoại",
    "cháu trai",
    "cháu gái",
    "chú ( người)",
    "bạn trai",
    "bạn gái",
    "ai cho",
    "ai bảo",
    "thích thú",
    "nghe",
    "nhìn",
    "hỏi thăm",
    "về 3 (đi về)",
    "ở ngoài",
    "ở trong",
    "không có",
    "vui sướng",
    "mở cửa",
    "quên",
    "ghi nhớ",
    "đón",
    "đợi",
    "sáng",
    "hàng ngày",
    "mỗi ngày",
    "tháng",
    "tuần",
    "tuần này",
    "tuần sau",
    "tuần trước",
]

PRESETS = {
    "basic": CORE_DAILY_CLASSES,
    "standard": STANDARD_DAILY_CLASSES,
    "extended": EXTENDED_DAILY_CLASSES,
}


def filter_and_save_preset(
    preset_name: str,
    selected_classes: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    """Filter metadata splits for a given preset, re-index labels, and generate vocab & config."""
    # Deduplicate while preserving deterministic order
    sorted_classes = sorted(list(set(selected_classes)))
    label_map = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}

    metadata_dir = ROOT / "dataset" / "metadata"
    artifacts_dir = ROOT / "artifacts"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save vocab JSON
    vocab_tokens = ["<blank>", "<unk>"] + sorted_classes
    vocab_file_name = f"vocab_{preset_name}.json"
    vocab_path = artifacts_dir / vocab_file_name
    vocab_path.write_text(
        json.dumps({"tokens": vocab_tokens}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2. Filter splits and re-index labels
    counts = {}
    split_dfs = {"train": train_df, "val": val_df, "test": test_df}

    for split_name, df in split_dfs.items():
        filtered = df[df["text"].isin(label_map)].copy()
        # Re-index label according to new compact vocabulary
        filtered["label"] = filtered["text"].map(label_map)
        filtered["token_text"] = filtered["text"]
        filtered.sort_values(by=["label", "feature_path"], inplace=True)

        # Save preset-specific CSV
        out_csv = metadata_dir / f"{split_name}_{preset_name}.csv"
        filtered.to_csv(out_csv, index=False, encoding="utf-8")
        counts[split_name] = len(filtered)

    # 3. Generate matching config YAML
    config_source = ROOT / "config.yaml"
    if config_source.is_file():
        with open(config_source, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    cfg["data"]["train_metadata"] = f"dataset/metadata/train_{preset_name}.csv"
    cfg["data"]["val_metadata"] = f"dataset/metadata/val_{preset_name}.csv"
    cfg["data"]["test_metadata"] = f"dataset/metadata/test_{preset_name}.csv"
    if "paths" not in cfg:
        cfg["paths"] = {}
    cfg["paths"]["vocab_file"] = f"artifacts/{vocab_file_name}"
    cfg["paths"]["checkpoint_dir"] = f"checkpoints_{preset_name}"
    cfg["model"]["num_classes"] = len(vocab_tokens)

    preset_config_file = ROOT / f"config_{preset_name}.yaml"
    with open(preset_config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    return {
        "classes": len(sorted_classes),
        "train": counts["train"],
        "val": counts["val"],
        "test": counts["test"],
        "total": counts["train"] + counts["val"] + counts["test"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["basic", "standard", "extended", "all"],
        default="all",
        help="Preset to generate: basic (~47 classes), standard (~116 classes), extended (~169 classes), or all (default: all)",
    )
    args = parser.parse_args()

    meta_dir = ROOT / "dataset" / "metadata"
    train_csv = meta_dir / "train.csv"
    val_csv = meta_dir / "val.csv"
    test_csv = meta_dir / "test.csv"

    if not (train_csv.is_file() and val_csv.is_file() and test_csv.is_file()):
        print(f"Error: Original metadata files not found in {meta_dir}", file=sys.stderr)
        sys.exit(1)

    print("Loading source datasets...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    print(f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples across {train_df['text'].nunique()} classes.")

    presets_to_run = list(PRESETS.keys()) if args.preset == "all" else [args.preset]

    print("\nFiltering and generating basic datasets:")
    print("=" * 65)
    for p_name in presets_to_run:
        classes = PRESETS[p_name]
        res = filter_and_save_preset(
            p_name, classes, train_df, val_df, test_df
        )
        tag = " [PRIMARY -> config_basic.yaml]" if p_name == "basic" else ""
        print(
            f"Preset '{p_name}'{tag}:\n"
            f"  - Classes: {res['classes']}\n"
            f"  - Train samples: {res['train']}\n"
            f"  - Val samples: {res['val']}\n"
            f"  - Test samples: {res['test']}\n"
            f"  - Total samples: {res['total']} (Avg {res['total']/res['classes']:.1f} clips/class)\n"
        )

    print("=" * 65)
    print("Successfully created lightweight basic datasets!")
    print(f"Primary basic files created at:")
    print(f"  - dataset/metadata/train_basic.csv")
    print(f"  - dataset/metadata/val_basic.csv")
    print(f"  - dataset/metadata/test_basic.csv")
    print(f"  - artifacts/vocab_basic.json")
    print(f"  - config_basic.yaml")
    print("\nTo train on the basic dataset:")
    print("  .venv\\Scripts\\python.exe train.py --config config_basic.yaml --device auto")


if __name__ == "__main__":
    main()
