# Project Report

## 1. Problem

Build a local continuous video-to-sign-token prototype that observes a webcam stream and emits a transcript without a capture action for every sign.

## 2. Motivation

Landmark sequences are smaller than raw video and retain hand/body motion. A temporal model with CTC can learn from video-level transcripts without frame-aligned labels.

## 3. System architecture

OpenCV reads BGR frames directly from the computer's webcam. MediaPipe Holistic extracts both hands, pose, and optional selected face points. The pipeline normalizes each group, appends validity masks, buffers a sequence, runs a Transformer encoder, and greedily decodes CTC output. The desktop preview renders Vietnamese text locally; optional speech uses local system voices. No browser or web server is needed.

## 4. Dataset

The workspace contains the CC BY 4.0 VSL 100-class isolated-word training set (3,875 source clips; 2,806 unique recordings after exact-duplicate removal). The current feature split contains 1,954 train, 441 validation, and 409 test clips; two clips without detectable landmarks were excluded. A trained checkpoint is available. The corpus does not provide continuous sentence translations or signer IDs, so this experiment is neither a sentence-translation benchmark nor signer-independent. ViSL-News remains a possible sentence-level research source but its source videos are not redistributed and its card describes non-commercial research. VSL400 offers isolated-word glosses under controlled access. See `DATASET_SETUP.md` for provenance and preparation steps.

## 5. Preprocessing

The converter expects `video_path,text,language` and an optional `subject_id`. It groups whole subjects for train/validation/test splits, falling back to whole-video groups. Feature sequences are stored as `.npy` and referenced from split CSV files. The extractor honors only confidence fields actually set by MediaPipe and excludes clips with no detected landmarks; the data loader rejects empty transcripts, invalid dimensions, NaN values, and missing files.

## 6. Feature extraction

Each hand has 21 points, pose has 33, and the optional face subset has 17 by default. Hands are wrist-centered and scaled by wrist-to-middle-MCP distance. Pose uses shoulder landmarks as reference/scale anchors; unavailable anchors fall back to visible-point statistics. Every point stores normalized XYZ plus a validity mask. Missing groups produce zero-valued features.

## 7. Transformer

The implemented model projects `[B,T,F]` landmark inputs to `d_model`, adds sinusoidal positions, applies configurable Transformer encoder layers, and classifies each frame into vocabulary symbols plus CTC blank.

## 8. CTC

Training uses PyTorch `CTCLoss` over complete transcript token sequences. Greedy decoding removes blanks and collapses adjacent repeated frame labels. The dataset does not require per-frame labels.

## 9. Training

Training supports CUDA/CPU selection, AMP on CUDA, gradient clipping, AdamW, a plateau scheduler, early stopping, CSV metrics, and best/last checkpoints. Landmark augmentation is configurable. The VSL100 run used the RTX 5050 Laptop GPU and stopped after 31 epochs; the best checkpoint was epoch 23.

## 10. Evaluation

The evaluator reports CTC loss, character error rate, word error rate, and sentence accuracy, and saves JSON and plots. On 409 held-out clips, the best checkpoint reached test loss 0.6801, CER 13.34%, WER 14.95%, and exact word accuracy 87.53%. Signer IDs are unavailable, so these results are not signer-independent and do not validate continuous VSL translation.

## 11. Realtime inference

The desktop OpenCV loop processes frames continuously, keeps an overlapping sequence buffer, runs the model every configured stride on a background worker, applies confidence filtering and temporal smoothing, and detects low motion after a configurable silence interval. Camera frames remain in memory and are not persisted. Vietnamese is rendered with a Unicode font so diacritics display correctly.

## 12. Translation

The current postprocessor normalizes casing/punctuation and has one explicit name-introduction rewrite. Custom phrase-reordering rules are supported but empty by default; invalid rules are not implied by the VSL100 class vocabulary. This is text formatting, not a trained semantic translation model.

## 13. Text-to-Speech

`pyttsx3` accesses local system voices without an API key. Whether Vietnamese speech is available depends on the Windows voice packages installed by the user.

## 14. Limitations

The software pipeline and model tests pass, and this workspace now has a learned isolated-word checkpoint. Current word-level CTC, limited translation rules, five-signer candidate sentence corpus, camera conditions, and lack of multi-person selection limit expected use. Signer-independent and continuous-sentence performance remain unvalidated.

## 15. Future development

Acquire consented licensed VSL data, validate gloss/transcript conventions with Deaf signers, run signer-independent experiments, train and report real CER/WER, improve continuous utterance segmentation, assess actual webcam latency, and replace the narrow formatter with a locally evaluated translation model.
