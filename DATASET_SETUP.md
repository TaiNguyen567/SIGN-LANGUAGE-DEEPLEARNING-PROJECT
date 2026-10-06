# Dataset Setup

## Current state

This checkout contains the VSL 100-class training archive at `dataset/raw/vsl100/dataset.zip`. The archive is excluded from version control. The importer extracts only labeled training clips, removes byte-identical copies, and writes `dataset/metadata/vsl100.csv`; the resulting source set has 2,806 unique clips across 100 isolated-word classes. The prepared metadata currently has 1,954 train, 441 validation, and 409 test clips; two source clips with no detectable landmarks were excluded. This workspace also has cached features and a trained checkpoint, though generated artifacts may not be present in a clean checkout.

The latest held-out result is 87.53% exact word accuracy (sentence accuracy for one-token clips), 14.95% WER, and 13.34% CER on 409 samples, using the checkpoint from epoch 23. Since the source has no signer IDs, this is not signer-independent evaluation.

This data supports isolated VSL word/class recognition only. It is not a continuous VSL sentence corpus and does not establish sentence translation performance. The archive also contains public/private test videos; the importer deliberately does not extract those unlabeled test splits.

## Included VSL 100-class dataset

Source: [ViSignLanguage-Video dataset card](https://huggingface.co/datasets/star092304/ViSignLanguage-Video), which identifies the source as the PTIT AI Challenge CV dataset and declares CC BY 4.0. Keep attribution to the dataset/source and the license link when using it. The source card reports 3,875 labeled training clips across 100 classes in this release. The archive contains 1,069 additional copies that are byte-identical to another clip; the importer keeps one copy per original recording ID. Some labels in the source CSV have character-encoding damage, so training labels are read from the normalized class-directory names instead.

The archive's test folders are not used. Signer IDs are not provided, so stratifying by class and grouping duplicate recording IDs prevents known clip-level leakage but does not provide signer-independent evaluation. Treat results as a small word-classification experiment, not a VSL translation benchmark.

The archive is pinned by SHA-256 `cdfec1080e8b75cf10a972a4d7cf54fcfa4bf1b874002d28eae8405188a2a308`. To recreate the import from a clean checkout, download `dataset.zip` from the source page into `dataset/raw/vsl100/`, then run:

```powershell
.venv\Scripts\python.exe scripts\import_vsl100.py
.venv\Scripts\python.exe scripts\prepare_dataset.py --metadata dataset\metadata\vsl100.csv --config config.yaml --stratify-by token_text
.venv\Scripts\python.exe train.py --config config.yaml --device auto
```

The imported manifest keeps display labels in `text` and a single CTC class token in `token_text`, so multiword labels such as “Bệnh nhân” remain one class. The preparation command splits each class into train/validation/test and extracts cached MediaPipe landmarks. The included archive itself is already imported in this workspace; rerun the first command only to verify or rebuild the derived videos and metadata.

## Candidate sources

### ViSL-News

Source: [ViSL-News dataset card](https://huggingface.co/datasets/kha2612/ViSL-News)

ViSL-News is the closer research match for sentence-level continuous VSL-to-Vietnamese work. Its dataset card describes 26,395 sentence clips, about 67 hours, five signers, and Vietnamese sentence transcripts. The source is sign-interpreted HTV news, so it does not represent everyday conversation. Its published split includes all five signers in train, validation, and test; use a signer-held-out split for a subject-independent evaluation.

The dataset card describes the release as non-commercial research data. It says the original broadcasts are not redistributed, rights remain with their owners, and users are responsible for source-platform terms and copyright. The card does not state a clear open data license for the source videos. Do not download or redistribute those videos unless the dataset maintainers and original rights holders permit it. The metadata/release may be inspected separately from source media; check the current card and terms before use.

### VSL400

Source: [VSL400 Zenodo record](https://zenodo.org/records/17943574)

VSL400 contains 74,259 manually annotated isolated clips across 400 glosses, performed by 28 signers and recorded from three views. It is useful for word-level VSL recognition experiments, but it is not a continuous sentence-translation corpus. Zenodo marks the files as restricted; video access requires the VSL400 Data Usage Agreement. The record currently reports about 14.1 TB across its files. Request access and read the agreement before planning storage or training.

ViSL-News and VSL400 remain separate candidate sources with their own access conditions. Do not merge them with this isolated-word dataset without checking label conventions, splits, and rights.

## Accepted input format

For the included VSL100 archive, use `dataset/metadata/vsl100.csv` generated by `scripts/import_vsl100.py`. For another licensed collection, create a UTF-8 CSV with these columns:

```csv
video_path,text,language,subject_id
dataset/videos/clip_001.mp4,TÔI MUỐN UỐNG NƯỚC,vi,signer_01
dataset/videos/clip_002.mp4,XIN CHÀO,vi,signer_02
```

`video_path` may be absolute or relative to the project root. `text` is the display transcript for the complete clip. `token_text` is optional and can represent an atomic class token when a label contains multiple words. `language` defaults to `vi`. `subject_id` is strongly recommended; records are grouped by subject before train/validation/test splitting. Otherwise, whole `split_group` values or videos are kept together. Never split extracted frames or duplicate clips from one source recording across partitions.

Store licensed local videos under `dataset/videos/`. Do not commit source videos, credentials, or private recordings. Confirm participant consent and dataset terms before extracting landmarks.

## Prepare and train

```powershell
.venv\Scripts\python.exe scripts\prepare_dataset.py --metadata dataset\metadata\all.csv --config config.yaml
.venv\Scripts\python.exe train.py --config config.yaml --device auto
.venv\Scripts\python.exe evaluate.py --config config.yaml --checkpoint checkpoints\best_model.pt --device auto
```

Preparation splits by subject/group when those columns are present, can stratify by a label column, extracts MediaPipe landmarks incrementally, and saves one `.npy` feature sequence per video under `features/train`, `features/val`, or `features/test`. It writes the matching CSV files in `dataset/metadata/`. Cached features avoid rerunning MediaPipe every epoch.

Inspect the resulting split metadata before training. The split tool checks that a video or known subject does not occur across partitions. For an already partitioned source dataset, create a combined metadata CSV only after deciding on a leakage-safe signer split.
