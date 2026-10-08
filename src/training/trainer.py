"""CTC training loop with validation, checkpoints, and early stopping."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.tokenizer import SignTokenizer
from src.evaluation.metrics import character_error_rate, sentence_accuracy, word_error_rate
from src.models.ctc_decoder import CTCGreedyDecoder
from src.training.losses import CTCLoss


class Trainer:
    """Own optimizer state and execute transcript-level CTC training epochs."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: SignTokenizer,
        *,
        device: torch.device,
        config: dict[str, Any],
        checkpoint_dir: str | Path = "checkpoints",
        log_dir: str | Path = "logs",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        training = config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training.get("learning_rate", 1e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=int(training.get("lr_scheduler_patience", 2)),
        )
        self.criterion = CTCLoss(blank_id=tokenizer.blank_id)
        self.decoder = CTCGreedyDecoder(blank_id=tokenizer.blank_id)
        self.amp_enabled = bool(training.get("mixed_precision", True) and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.max_grad_norm = float(training.get("gradient_clip_norm", 1.0))
        self.logger = logging.getLogger("sign_language_ai.training")
        self.csv_path = self.log_dir / "training.csv"

    def resume_from_checkpoint(self, checkpoint_path: Path, override_lr: float | None = None) -> tuple[int, float]:
        """Restore model, optimizer, and scheduler states from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if override_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = float(override_lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=int(self.config.get("training", {}).get("lr_scheduler_patience", 2)),
            )
        elif "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.logger.info("Resumed checkpoint %s at epoch %d (best_val_loss=%.4f, lr=%.2e)", checkpoint_path.name, start_epoch, best_val_loss, self.optimizer.param_groups[0]["lr"])
        return start_epoch, best_val_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 8,
        start_epoch: int = 1,
        initial_best_val_loss: float = float("inf"),
    ) -> list[dict[str, float]]:
        if epochs < 1 or patience < 1:
            raise ValueError("epochs and early-stopping patience must be positive")
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("Training and validation datasets must both contain at least one batch")
        if start_epoch <= 1:
            self.csv_path.unlink(missing_ok=True)
        history: list[dict[str, float]] = []
        best_val_loss = initial_best_val_loss
        best_epoch = start_epoch - 1 if initial_best_val_loss < float("inf") else 0
        stale_epochs = 0
        for epoch in range(start_epoch, epochs + 1):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            train_result = self._run_epoch(train_loader, training=True)
            val_result = self._run_epoch(val_loader, training=False)
            self.scheduler.step(val_result["loss"])
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            row = {
                "epoch": float(epoch),
                "train_loss": train_result["loss"],
                "val_loss": val_result["loss"],
                "cer": val_result["cer"],
                "wer": val_result["wer"],
                "sentence_accuracy": val_result["sentence_accuracy"],
                "learning_rate": learning_rate,
                "gpu_memory_gib": (
                    torch.cuda.max_memory_allocated(self.device) / 1024**3 if self.device.type == "cuda" else 0.0
                ),
            }
            history.append(row)
            self._append_csv(row)
            self.logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f CER=%.4f WER=%.4f acc=%.2f%% lr=%.2e",
                epoch, row["train_loss"], row["val_loss"], row["cer"], row["wer"], row["sentence_accuracy"] * 100.0, learning_rate,
            )

            improved = val_result["loss"] < best_val_loss
            if improved:
                best_val_loss = val_result["loss"]
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            self._save_checkpoint(self.checkpoint_dir / "last_model.pt", epoch, best_val_loss)
            if improved:
                self._save_checkpoint(self.checkpoint_dir / "best_model.pt", epoch, best_val_loss)
            if stale_epochs >= patience:
                self.logger.info("early stopping at epoch %d; best epoch was %d", epoch, best_epoch)
                break
        return history

    def _run_epoch(self, loader: DataLoader, *, training: bool) -> dict[str, float]:
        self.model.train(training)
        loss_total = torch.zeros((), dtype=torch.float32, device=self.device)
        sample_count = 0
        references: list[str] = []
        hypotheses: list[str] = []
        iterator = tqdm(loader, leave=False, disable=True, desc="train" if training else "validation")
        for step, batch in enumerate(iterator):
            features = batch["features"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    log_probs = self.model(features, input_lengths)
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if (step + 1) % 200 == 0 or (step + 1) == len(loader):
                        self.logger.info("train step %d/%d (%.1f%%) batch_loss=%.4f", step + 1, len(loader), (step + 1) * 100.0 / len(loader), loss.item())

            batch_size = features.shape[0]
            loss_total.add_(loss.detach().float() * batch_size)
            sample_count += batch_size
            if not training:
                decoded = self.decoder.decode(log_probs, input_lengths)
                cursor = 0
                for target_length, output in zip(target_lengths.tolist(), decoded):
                    target_ids = targets[cursor:cursor + target_length].tolist()
                    cursor += target_length
                    references.append(self.tokenizer.decode(target_ids))
                    hypotheses.append(self.tokenizer.decode(output.token_ids))

        average_loss = float(loss_total.item()) / max(sample_count, 1)
        return {
            "loss": average_loss,
            "cer": character_error_rate(references, hypotheses) if references else 0.0,
            "wer": word_error_rate(references, hypotheses) if references else 0.0,
            "sentence_accuracy": sentence_accuracy(references, hypotheses) if references else 0.0,
        }

    def _save_checkpoint(self, path: Path, epoch: int, best_val_loss: float) -> None:
        model_config = {
            "input_dim": self.model.input_dim,
            "num_classes": self.model.num_classes,
            "d_model": self.model.projection[0].out_features,
            "nhead": self.model.encoder.layers[0].self_attn.num_heads,
            "num_layers": len(self.model.encoder.layers),
            "dim_feedforward": self.model.encoder.layers[0].linear1.out_features,
            "dropout": self.model.encoder.layers[0].dropout.p,
        }
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "vocab": self.tokenizer.tokens,
            "config": self.config,
            "model_config": model_config,
            "feature_dim": self.model.input_dim,
        }, path)

    def _append_csv(self, row: dict[str, float]) -> None:
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
