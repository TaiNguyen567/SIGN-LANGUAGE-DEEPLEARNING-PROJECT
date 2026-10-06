"""Train the landmark Transformer with transcript-level CTC supervision."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.augmentation import LandmarkAugmenter
from src.data.dataset import SignLanguageDataset, ctc_collate_fn
from src.data.tokenizer import SignTokenizer
from src.features.landmark_extractor import LandmarkExtractor
from src.models.transformer import SignLanguageTransformer
from src.training.trainer import Trainer
from src.utils.config import feature_config_from_dict, load_config, resolve_project_path
from src.utils.device import format_device_info, get_device, get_device_info
from src.utils.logger import configure_logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    try:
        config = load_config(resolve_project_path(args.config, project_root=ROOT))
        paths = config["paths"]
        metadata = resolve_project_path(config["data"]["train_metadata"], project_root=ROOT)
        validation_metadata = resolve_project_path(config["data"]["val_metadata"], project_root=ROOT)
        if not metadata.is_file() or not validation_metadata.is_file():
            raise FileNotFoundError(
                "Training/validation metadata is missing. Add real videos and transcripts, then run "
                "scripts/prepare_dataset.py; see DATASET_SETUP.md. No demo data is used for training."
            )
        train_frame = pd.read_csv(metadata, keep_default_na=False)
        if train_frame.empty:
            raise ValueError(f"Training dataset is empty: {metadata}")
        token_column = train_frame["token_text"] if "token_text" in train_frame.columns else train_frame["text"]
        tokenizer = SignTokenizer.build(token_column.astype(str))
        vocab_path = resolve_project_path("artifacts/vocab.json", project_root=ROOT)
        tokenizer.save(vocab_path)

        seed = int(config.get("project", {}).get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = get_device(args.device)
        feature_config = feature_config_from_dict(config.get("features", {}))
        model_config = config.get("model", {})
        model = SignLanguageTransformer(
            input_dim=feature_config.feature_dim,
            num_classes=tokenizer.vocabulary_size,
            d_model=int(model_config.get("d_model", 256)),
            nhead=int(model_config.get("nhead", 8)),
            num_layers=int(model_config.get("num_layers", 4)),
            dim_feedforward=int(model_config.get("dim_feedforward", 1024)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
        augmentation_config = config.get("augmentation", {})
        train_set = SignLanguageDataset(
            metadata, tokenizer, project_root=ROOT, feature_config=feature_config,
            augmenter=LandmarkAugmenter(augmentation_config, seed=seed) if augmentation_config.get("enabled") else None,
        )
        val_set = SignLanguageDataset(validation_metadata, tokenizer, project_root=ROOT, feature_config=feature_config)
        training = config.get("training", {})
        workers = int(training.get("num_workers", 0))
        common_loader_args = {
            "batch_size": int(training.get("batch_size", 8)),
            "num_workers": workers,
            "collate_fn": ctc_collate_fn,
            "pin_memory": device.type == "cuda",
            "persistent_workers": workers > 0,
        }
        train_loader = DataLoader(train_set, shuffle=True, **common_loader_args)
        val_loader = DataLoader(val_set, shuffle=False, **common_loader_args)
        log_dir = resolve_project_path(paths.get("log_dir", "logs"), project_root=ROOT)
        logger = configure_logger("sign_language_ai.training", log_dir / "train.log", console=True)
        info = get_device_info(device)
        logger.info("device=%s feature_dim=%d vocabulary=%d", format_device_info(info), feature_config.feature_dim, tokenizer.vocabulary_size)
        trainer = Trainer(
            model, tokenizer, device=device, config=config,
            checkpoint_dir=resolve_project_path(paths.get("checkpoint_dir", "checkpoints"), project_root=ROOT),
            log_dir=log_dir,
        )
        history = trainer.fit(
            train_loader, val_loader,
            epochs=int(training.get("epochs", 50)),
            patience=int(training.get("early_stopping_patience", 8)),
        )
        print(f"Training finished after {len(history)} epochs. Best checkpoint: checkpoints/best_model.pt")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Training could not start: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
