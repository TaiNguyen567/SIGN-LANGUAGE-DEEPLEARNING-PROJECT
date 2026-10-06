import torch
from torch.utils.data import DataLoader

from src.data.dataset import ctc_collate_fn
from src.data.tokenizer import SignTokenizer
from src.models.transformer import SignLanguageTransformer
from src.training.trainer import Trainer
from src.utils.checkpoint import load_model_checkpoint


def test_training_writes_loadable_checkpoint(tmp_path):
    tokenizer = SignTokenizer(["<blank>", "<unk>", "TÔI", "MUỐN"])
    samples = [{
        "features": torch.randn(6, 4),
        "target": torch.tensor([2, 3]),
        "text": "TÔI MUỐN",
        "video_path": "sample.mp4",
    }]
    loader = DataLoader(samples, batch_size=1, collate_fn=ctc_collate_fn)
    model = SignLanguageTransformer(
        4, tokenizer.vocabulary_size, d_model=8, nhead=2,
        num_layers=1, dim_feedforward=16, dropout=0.0, max_len=8,
    )
    model_config = {
        "d_model": 8, "nhead": 2, "num_layers": 1,
        "dim_feedforward": 16, "dropout": 0.0,
    }
    trainer = Trainer(
        model, tokenizer, device=torch.device("cpu"),
        config={"training": {"learning_rate": 0.001, "mixed_precision": False}},
        checkpoint_dir=tmp_path / "checkpoints", log_dir=tmp_path / "logs",
    )
    trainer.csv_path.write_text("stale run history\n", encoding="utf-8")

    history = trainer.fit(loader, loader, epochs=1, patience=1)
    restored, restored_tokenizer, checkpoint = load_model_checkpoint(
        tmp_path / "checkpoints" / "best_model.pt",
        device=torch.device("cpu"), expected_feature_dim=4, expected_model_config=model_config,
    )

    assert len(history) == 1
    assert (tmp_path / "checkpoints" / "last_model.pt").is_file()
    assert restored.input_dim == 4
    assert restored_tokenizer.tokens == tokenizer.tokens
    assert checkpoint["epoch"] == 1
    assert "stale run history" not in trainer.csv_path.read_text(encoding="utf-8")
