import numpy as np
import pandas as pd

from src.data.dataset import SignLanguageDataset, ctc_collate_fn
from src.data.preprocessing import split_records, validate_splits
from src.data.tokenizer import SignTokenizer


def test_dataset_loads_cached_features_and_ctc_batch(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.save(feature_dir / "one.npy", np.ones((4, 12), dtype=np.float32))
    np.save(feature_dir / "two.npy", np.full((6, 12), 2.0, dtype=np.float32))
    metadata = tmp_path / "train.csv"
    pd.DataFrame([
        {"video_path": "videos/one.mp4", "feature_path": "features/one.npy", "text": "TÔI MUỐN"},
        {"video_path": "videos/two.mp4", "feature_path": "features/two.npy", "text": "XIN CHÀO"},
    ]).to_csv(metadata, index=False)
    tokenizer = SignTokenizer.build(["TÔI MUỐN", "XIN CHÀO"])
    dataset = SignLanguageDataset(metadata, tokenizer, project_root=tmp_path)

    batch = ctc_collate_fn([dataset[0], dataset[1]])

    assert len(dataset) == 2
    assert batch["features"].shape == (2, 6, 12)
    assert batch["input_lengths"].tolist() == [4, 6]
    assert batch["target_lengths"].tolist() == [2, 2]
    assert batch["targets"].numel() == 4


def test_class_token_text_keeps_multiword_vsl_labels_atomic(tmp_path):
    feature_path = tmp_path / "one.npy"
    np.save(feature_path, np.ones((4, 12), dtype=np.float32))
    metadata = tmp_path / "train.csv"
    pd.DataFrame([{
        "video_path": "videos/one.mp4",
        "feature_path": "one.npy",
        "text": "Bệnh nhân",
        "token_text": "Bệnh_nhân",
    }]).to_csv(metadata, index=False)
    tokenizer = SignTokenizer.build(["Bệnh_nhân"])
    dataset = SignLanguageDataset(metadata, tokenizer, project_root=tmp_path)

    sample = dataset[0]

    assert sample["text"] == "Bệnh nhân"
    assert sample["target"].numel() == 1
    assert tokenizer.decode(sample["target"].tolist()) == "BỆNH NHÂN"


def test_split_groups_subjects_and_checks_leakage():
    records = [
        {"video_path": f"video-{subject}-{i}.mp4", "subject_id": subject}
        for subject in range(10) for i in range(2)
    ]

    splits = split_records(records, seed=7)

    assert sum(map(len, splits.values())) == len(records)
    validate_splits(splits)
    assert len(splits["train"]) > len(splits["val"])
