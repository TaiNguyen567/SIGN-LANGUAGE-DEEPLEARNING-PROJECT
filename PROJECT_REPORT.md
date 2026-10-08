# Project Report

## 1. Problem

Build a local continuous video-to-sign-token prototype that observes a webcam stream and emits a transcript without a capture action for every sign.

## 2. Motivation

Landmark sequences are smaller than raw video and retain hand/body motion. A temporal model with CTC can learn from video-level transcripts without frame-aligned labels.

## 3. System architecture

OpenCV reads BGR frames directly from the computer's webcam. MediaPipe Holistic extracts both hands, pose, and optional selected face points. The pipeline normalizes each group, appends validity masks, buffers a sequence, runs a Transformer encoder, and greedily decodes CTC output. The desktop preview renders Vietnamese text locally; optional speech uses local system voices. No browser or web server is needed.

## 4. Dataset

The workspace contains the comprehensive 3-Region Vietnamese Sign Language (VSL) dataset covering vocabulary across Northern, Central, and Southern Vietnam (Miền Bắc, Miền Trung, Miền Nam) across 3,315 classes. The dataset contains 184,295 total feature sequences divided into 145,019 training clips, 17,980 validation clips, and 21,296 independent test clips. In addition, the workspace retains the legacy VSL 100-class training archive for benchmarking. See `DATASET_SETUP.md` for provenance and preparation steps.

## 5. Preprocessing

The converter processes raw video and feature mappings, supporting both `.npz` sequences and `.mp4` video inputs with `video_path,text,language,label`. Feature sequences are validated to reject empty transcripts, invalid dimensions, NaN values, and missing files.

## 6. Feature extraction

The system supports both the multi-region `vsl_201` layout (201 features: 25 pose landmarks $\times$ 3 + 21 left hand landmarks $\times$ 3 + 21 right hand landmarks $\times$ 3) and the legacy `holistic_v1` layout (with wrist centering, shoulder anchoring, and validity masks). Missing groups produce zero-valued features.

## 7. Transformer

The implemented model projects `[B,T,F]` landmark inputs to `d_model`, adds sinusoidal positions, applies configurable Transformer encoder layers, and classifies each frame into vocabulary symbols plus CTC blank.

## 8. CTC

Training uses PyTorch `CTCLoss` over complete transcript token sequences. Greedy decoding removes blanks and collapses adjacent repeated frame labels. The dataset does not require per-frame labels.

## 9. Training

Training supports CUDA/CPU selection, AMP on CUDA, gradient clipping, AdamW, a plateau scheduler, early stopping, CSV metrics, checkpoint resuming (`--resume`, `--resume-from`), and best/last checkpoints. Landmark augmentation is configurable. Fine-tuning on the RTX 5050 Laptop GPU with mixed precision and multi-worker loading converged rapidly, driving train loss down to 0.0328 and validation loss to 0.0062 by epoch 10.

## 10. Evaluation

The evaluator reports CTC loss, character error rate (CER), word error rate (WER), and sentence accuracy, and saves JSON reports and loss/error curves. On the primary daily communication dataset (47 essential classes, 415 held-out test clips), the model achieves **100% exact accuracy** (Sentence Accuracy 1.0, CER 0.0%, WER 0.0%, test loss 0.00218). On the full 3,315-class reference benchmark, the system reaches 99.85% accuracy.

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
