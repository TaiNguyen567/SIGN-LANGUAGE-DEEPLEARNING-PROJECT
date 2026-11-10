"""Detailed per-class accuracy evaluator and single-video tester for Sign Language AI."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import SignLanguageDataset
from src.features.landmark_extractor import LandmarkExtractor
from src.models.ctc_decoder import CTCGreedyDecoder
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config import feature_config_from_dict, load_config, resolve_project_path
from src.utils.device import get_device


def evaluate_dataset(
    checkpoint_path: Path,
    config_path: Path,
    split: str = "test",
    device_name: str = "auto",
) -> int:
    config = load_config(config_path)
    device = get_device(device_name)
    feature_config = feature_config_from_dict(config.get("features", {}))
    model, tokenizer, ckpt = load_model_checkpoint(
        checkpoint_path,
        device=device,
        expected_feature_dim=feature_config.feature_dim,
        expected_model_config=config.get("model", {}),
    )
    meta_key = f"{split}_metadata"
    csv_rel = config["data"].get(meta_key, f"dataset/metadata/{split}.csv")
    csv_path = resolve_project_path(csv_rel, project_root=ROOT)
    dataset = SignLanguageDataset(csv_path, tokenizer, project_root=ROOT, feature_config=feature_config)
    decoder = CTCGreedyDecoder(blank_id=tokenizer.blank_id)

    stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "conf_sum": 0.0, "mistakes": []}
    )
    total_samples = len(dataset)
    total_correct = 0

    model.eval()
    with torch.inference_mode():
        for idx in range(total_samples):
            item = dataset[idx]
            features = item["features"].unsqueeze(0).to(device)
            lengths = torch.tensor([item["features"].shape[0]], dtype=torch.long)
            log_probs = model(features, lengths)
            decoded = decoder.decode(log_probs, lengths)[0]
            pred_text = tokenizer.decode(decoded.token_ids)
            true_text = str(item["text"])

            entry = stats[true_text]
            entry["total"] = int(entry["total"]) + 1
            entry["conf_sum"] = float(entry["conf_sum"]) + float(decoded.confidence)
            if pred_text == true_text:
                entry["correct"] = int(entry["correct"]) + 1
                total_correct += 1
            else:
                mistakes = list(entry["mistakes"])
                mistakes.append(f"{pred_text or '<trống>'} ({decoded.confidence:.0%})")
                entry["mistakes"] = mistakes

    print("=" * 88)
    print(f" BẢNG KIỂM TRA ĐỘ CHÍNH XÁC TỪNG TỪ TRÊN TẬP {split.upper()} ({total_samples} video độc lập)")
    print(f" Checkpoint: {checkpoint_path.name} (Epoch {ckpt.get('epoch', '?')}) | Tổng số lớp: {len(stats)}")
    print("=" * 88)
    print(f"{'STT':<4} | {'Từ vựng (Ký hiệu)':<16} | {'Đúng/Tổng':<10} | {'Độ chính xác':<13} | {'Tin cậy TB':<10} | {'Nhầm sang (nếu có)'}")
    print("-" * 88)

    rows_for_csv = []
    for i, (word, info) in enumerate(sorted(stats.items(), key=lambda x: x[0]), start=1):
        tot = int(info["total"])
        cor = int(info["correct"])
        acc = (cor / tot * 100.0) if tot > 0 else 0.0
        avg_conf = (float(info["conf_sum"]) / tot * 100.0) if tot > 0 else 0.0
        mistakes_str = ", ".join(list(info["mistakes"])[:3])
        print(f"{i:<4} | {word:<16} | {cor:>3}/{tot:<6} | {acc:>10.1f}%  | {avg_conf:>8.1f}%  | {mistakes_str}")
        rows_for_csv.append({
            "word": word,
            "correct": cor,
            "total": tot,
            "accuracy_percent": round(acc, 2),
            "avg_confidence_percent": round(avg_conf, 2),
            "mistakes": mistakes_str,
        })

    overall_acc = (total_correct / max(1, total_samples)) * 100.0
    print("-" * 88)
    print(f" TỔNG KẾT: Đúng {total_correct}/{total_samples} video  =>  ĐỘ CHÍNH XÁC TRUNG BÌNH: {overall_acc:.2f}%")
    print("=" * 88)

    out_csv = ROOT / "reports" / "per_class_accuracy.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_for_csv).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Đã xuất bảng chi tiết ra file: {out_csv.relative_to(ROOT)}")
    return 0


def test_single_video(video_path: Path, checkpoint_path: Path, config_path: Path, device_name: str = "auto") -> int:
    config = load_config(config_path)
    device = get_device(device_name)
    feature_config = feature_config_from_dict(config.get("features", {}))
    model, tokenizer, _ = load_model_checkpoint(
        checkpoint_path,
        device=device,
        expected_feature_dim=feature_config.feature_dim,
        expected_model_config=config.get("model", {}),
    )
    decoder = CTCGreedyDecoder(blank_id=tokenizer.blank_id)
    with LandmarkExtractor(feature_config) as extractor:
        seq = extractor.extract_sequence(video_path)
    features = torch.from_numpy(seq).unsqueeze(0).to(device)
    lengths = torch.tensor([seq.shape[0]], dtype=torch.long)
    with torch.inference_mode():
        log_probs = model(features, lengths)
    decoded = decoder.decode(log_probs, lengths)[0]
    pred_text = tokenizer.decode(decoded.token_ids)
    probs = log_probs.detach().float().exp().squeeze(1)
    probs[:, 0] = 0.0
    probs[:, 1] = 0.0
    mean_probs = probs.mean(dim=0)
    top3_vals, top3_ids = torch.topk(mean_probs, k=min(3, mean_probs.numel()))
    print(f"\nVideo kiểm tra: {video_path}")
    print(f"-> Kết quả dự đoán: {pred_text!r} (Độ tin cậy đỉnh: {decoded.confidence:.1%})")
    print("-> Top 3 ứng viên gần nhất:")
    for rank, (tid, val) in enumerate(zip(top3_ids.tolist(), top3_vals.tolist()), start=1):
        print(f"   {rank}. {tokenizer.tokens[tid]} (điểm trung bình chuỗi: {val:.1%})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--video", type=str, default=None, help="Optional path to a single .mp4 video to test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ckpt = resolve_project_path(args.checkpoint, project_root=ROOT)
    cfg = resolve_project_path(args.config, project_root=ROOT)
    if args.video:
        return test_single_video(resolve_project_path(args.video, project_root=ROOT), ckpt, cfg, args.device)
    return evaluate_dataset(ckpt, cfg, args.split, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
