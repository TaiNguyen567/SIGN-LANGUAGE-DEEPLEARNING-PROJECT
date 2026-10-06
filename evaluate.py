"""Evaluate a trained checkpoint on a held-out video/subject split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import SignLanguageDataset, ctc_collate_fn
from src.evaluation.metrics import character_error_rate, sentence_accuracy, word_error_rate
from src.models.ctc_decoder import CTCGreedyDecoder
from src.training.losses import CTCLoss
from src.utils.checkpoint import load_model_checkpoint
from src.utils.config import feature_config_from_dict, load_config, resolve_project_path
from src.utils.device import get_device


def evaluate(checkpoint_path: Path, config_path: Path, metadata_path: Path | None, device_name: str) -> dict[str, float | int]:
    config = load_config(config_path)
    device = get_device(device_name)
    feature_config = feature_config_from_dict(config.get("features", {}))
    model, tokenizer, checkpoint = load_model_checkpoint(
        checkpoint_path,
        device=device,
        expected_feature_dim=feature_config.feature_dim,
        expected_model_config=config.get("model", {}),
    )
    test_csv = metadata_path or resolve_project_path(config["data"]["test_metadata"], project_root=ROOT)
    dataset = SignLanguageDataset(test_csv, tokenizer, project_root=ROOT, feature_config=feature_config)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
        collate_fn=ctc_collate_fn,
        pin_memory=device.type == "cuda",
    )
    criterion = CTCLoss(blank_id=tokenizer.blank_id)
    decoder = CTCGreedyDecoder(blank_id=tokenizer.blank_id)
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    sample_count = 0
    references: list[str] = []
    hypotheses: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            log_probs = model(features, batch["input_lengths"])
            loss = criterion(log_probs, targets, batch["input_lengths"], batch["target_lengths"])
            batch_size = features.shape[0]
            total_loss.add_(loss.detach().float() * batch_size)
            sample_count += batch_size
            decoded = decoder.decode(log_probs, batch["input_lengths"])
            cursor = 0
            for target_length, output in zip(batch["target_lengths"].tolist(), decoded):
                target_ids = targets[cursor:cursor + target_length].tolist()
                cursor += target_length
                references.append(tokenizer.decode(target_ids))
                hypotheses.append(tokenizer.decode(output.token_ids))

    metrics: dict[str, float | int] = {
        "test_loss": float(total_loss.item()) / max(sample_count, 1),
        "cer": character_error_rate(references, hypotheses),
        "wer": word_error_rate(references, hypotheses),
        "sentence_accuracy": sentence_accuracy(references, hypotheses),
        "samples": sample_count,
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
    }
    report_dir = resolve_project_path(config.get("paths", {}).get("report_dir", "reports"), project_root=ROOT)
    (report_dir / "plots").mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_training_history(resolve_project_path(config.get("paths", {}).get("log_dir", "logs"), project_root=ROOT), report_dir / "plots")
    return metrics


def _plot_training_history(log_dir: Path, plot_dir: Path) -> None:
    history_csv = log_dir / "training.csv"
    if not history_csv.is_file():
        return
    history = pd.read_csv(history_csv)
    if history.empty:
        return
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history["epoch"], history["train_loss"], label="Train loss")
    axis.plot(history["epoch"], history["val_loss"], label="Validation loss")
    axis.set(xlabel="Epoch", ylabel="CTC loss", title="Training loss")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "loss_curve.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history["epoch"], history["cer"], label="CER")
    axis.plot(history["epoch"], history["wer"], label="WER")
    axis.set(xlabel="Epoch", ylabel="Error rate", title="Validation error rates")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "wer_cer.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--metadata", help="Override the configured held-out test CSV")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        metrics = evaluate(
            resolve_project_path(args.checkpoint, project_root=ROOT),
            resolve_project_path(args.config, project_root=ROOT),
            resolve_project_path(args.metadata, project_root=ROOT) if args.metadata else None,
            args.device,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
