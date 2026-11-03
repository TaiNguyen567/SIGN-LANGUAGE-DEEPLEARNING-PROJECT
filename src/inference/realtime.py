"""Continuous webcam recognition over overlapping landmark sequences."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.data.tokenizer import SignTokenizer
from src.features.landmark_extractor import FeatureConfig, LandmarkExtractor
from src.inference.postprocess import confidence_filter, temporal_smoothing
from src.models.ctc_decoder import CTCGreedyDecoder
from src.translation.vietnamese import VietnamesePostProcessor


@dataclass(frozen=True)
class RealtimeFrame:
    image: np.ndarray
    sign_text: str
    translated_text: str
    confidence: float
    fps: float
    latency_ms: float
    silent: bool
    utterance_ended: bool = False
    completed_text: str = ""
    completed_translation: str = ""
    camera_fps: float = 0.0
    completed_events: tuple[tuple[str, str, float], ...] = ()
    cooldown_remaining: float = 0.0
    silence_remaining: float = 0.0
    gesture_elapsed: float = 0.0
    candidate_text: str = ""
    committed_text: str = ""
    error: str = ""


@dataclass(frozen=True)
class _FrameJob:
    frame: np.ndarray
    captured_at: float
    enqueued_at: float
    generation: int


@dataclass(frozen=True)
class _InferenceResult:
    generation: int
    sign_text: str
    translated_text: str
    confidence: float
    fps: float
    latency_ms: float
    silent: bool
    completed_text: str = ""
    completed_translation: str = ""
    cooldown_remaining: float = 0.0
    silence_remaining: float = 0.0
    gesture_elapsed: float = 0.0
    candidate_text: str = ""
    committed_text: str = ""
    error: str = ""


class RealtimeRecognizer:
    """Keep camera capture responsive while a worker processes landmark windows."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: SignTokenizer,
        *,
        device: torch.device,
        feature_config: FeatureConfig,
        config: dict[str, Any],
        extractor: LandmarkExtractor | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.device_label = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        self.config = config
        self.feature_config = feature_config
        self.extractor = extractor or LandmarkExtractor(feature_config)
        self.decoder = CTCGreedyDecoder(tokenizer.blank_id)
        translation_config = config.get("translation", {})
        self.translator = VietnamesePostProcessor.from_config(
            translation_config.get("reordering_rules"),
        )
        realtime = config.get("realtime", {})
        data = config.get("data", {})
        self.sequence_length = int(realtime.get("sequence_length", data.get("sequence_length", 30)))
        self.stride = max(1, int(realtime.get("prediction_stride", data.get("stride", 8))))
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.confidence_threshold = float(realtime.get("confidence_threshold", 0.85))
        self.silence_threshold = float(realtime.get("silence_threshold", 1.2))
        self.motion_threshold = float(realtime.get("motion_threshold", 0.015))
        self.process_every_n_frames = max(1, int(realtime.get("process_every_n_frames", 1)))
        self.show_landmarks = bool(realtime.get("show_landmarks", True))
        self.require_hand = bool(realtime.get("require_hand", False))
        self.smoothing_window = max(1, int(realtime.get("smoothing_window", 5)))
        self.smoothing_votes = max(1, int(realtime.get("smoothing_votes", 3)))
        self.cooldown_seconds = max(0.0, float(realtime.get("cooldown_seconds", 0.0)))
        self.single_word_per_turn = bool(realtime.get("single_word_per_turn", True))
        self.manual_confirm = bool(realtime.get("manual_confirm", False))
        self.transcript_file = Path(realtime.get("transcript_file", "logs/saved_transcript.txt"))
        self.frame_queue: Queue[_FrameJob] = Queue(maxsize=1)
        self.result_queue: Queue[_InferenceResult] = Queue()
        self._control_queue: Queue[int] = Queue()
        self._stop_event = Event()
        self._generation_lock = Lock()
        self._worker_lock = Lock()
        self._preview_lock = Lock()
        self._state_lock = Lock()
        self._worker: Thread | None = None
        self._closed = False
        self._generation = 0
        self._worker_generation = 0
        self._landmark_preview: np.ndarray | None = None
        self._latest_result: _InferenceResult | None = None
        self._completed_events: list[tuple[str, str, float]] = []
        self._last_capture_at: float | None = None
        self.camera_fps = 0.0
        self.dropped_frames = 0
        self.buffer: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.hypothesis_history: deque[list[str]] = deque(maxlen=self.smoothing_window)
        self.committed_tokens: list[str] = []
        self.pending_tokens: list[str] = []
        self._turn_scores: Counter[str] = Counter()
        self._turn_started_at: float | None = None
        self._cooldown_until: float = 0.0
        self._was_silent = False
        self.confidence = 0.0
        self._worker_confidence = 0.0
        self.last_motion_at: float | None = None
        self.previous_features: np.ndarray | None = None
        self.frame_count = 0
        self.last_model_frame = 0
        self.fps = 0.0
        self.last_frame_at: float | None = None
        self._display_tokens: tuple[str, ...] | None = None
        self._display_sign_text = ""
        self._display_translated_text = ""

    def confirm_current_sign(self) -> tuple[str, str, float] | None:
        """Confirm the current candidate sign (on Enter/Space) and append it to the saved sentence."""
        with self._state_lock:
            if not self.pending_tokens:
                return None
            confirmed_word = self.decode_tokens(self.pending_tokens)
            self.committed_tokens.extend(self.pending_tokens)
            self.pending_tokens.clear()
            self.hypothesis_history.clear()
            self._turn_scores.clear()
            self._turn_started_at = None
            self.buffer.clear()
            if self.cooldown_seconds > 0.0 and self.last_frame_at is not None:
                self._cooldown_until = self.last_frame_at + self.cooldown_seconds
            full_sentence = self.translator.translate(self.committed_tokens)
            conf = self._worker_confidence
            self._refresh_display_texts()
            event = (confirmed_word, full_sentence, conf)
            self._completed_events.append(event)
            self._save_transcript_to_disk(full_sentence)
            return event

    def pop_committed_sign(self) -> str:
        """Remove the most recently saved word (on Backspace) and return updated sentence."""
        with self._state_lock:
            if self.committed_tokens:
                self.committed_tokens.pop()
            self._refresh_display_texts()
            full_sentence = self.translator.translate(self.committed_tokens) if self.committed_tokens else ""
            self._save_transcript_to_disk(full_sentence)
            return full_sentence

    def _save_transcript_to_disk(self, sentence: str) -> None:
        try:
            self.transcript_file.parent.mkdir(parents=True, exist_ok=True)
            self.transcript_file.write_text(sentence + ("\n" if sentence else ""), encoding="utf-8")
        except OSError:
            pass

    def _refresh_display_texts(self) -> None:
        tokens = self.tokens
        self._display_tokens = tuple(tokens)
        self._display_sign_text = self.decode_tokens(tokens)
        self._display_translated_text = self.translator.translate(tokens) if tokens else ""

    def process_frame(self, frame: np.ndarray, now: float | None = None) -> RealtimeFrame:
        """Enqueue a camera frame and return a display frame without waiting for AI."""
        self.start_background()
        current_time = now if now is not None else time.monotonic()
        self.camera_fps = self._update_capture_fps(current_time)
        with self._generation_lock:
            generation = self._generation
        job = _FrameJob(frame.copy(), current_time, time.perf_counter(), generation)
        self._enqueue_latest(job)
        self._collect_results()

        latest = self._latest_result
        preview = frame.copy()
        if self.show_landmarks:
            with self._preview_lock:
                if self._landmark_preview is not None:
                    preview = self._landmark_preview.copy()
        with self._state_lock:
            candidate_text = self.decode_tokens(self.pending_tokens) if self.pending_tokens else ""
            committed_text = self.decode_tokens(self.committed_tokens) if self.committed_tokens else ""
            signs = self._display_sign_text if self._display_tokens is not None else (latest.sign_text if latest else "")
            if self.manual_confirm:
                translation = self.translator.translate(self.committed_tokens) if self.committed_tokens else ""
            else:
                translation = self._display_translated_text if self._display_tokens is not None else (latest.translated_text if latest else "")
        confidence = latest.confidence if latest else 0.0
        processing_fps = latest.fps if latest else 0.0
        latency_ms = latest.latency_ms if latest else 0.0
        silent = latest.silent if latest else False
        cooldown_remaining = latest.cooldown_remaining if latest else 0.0
        silence_remaining = latest.silence_remaining if latest else 0.0
        gesture_elapsed = latest.gesture_elapsed if latest else 0.0
        self._draw_status(
            preview,
            signs,
            translation,
            confidence,
            processing_fps,
            latency_ms,
            silent,
            cooldown_remaining=cooldown_remaining,
            silence_remaining=silence_remaining,
            gesture_elapsed=gesture_elapsed,
            candidate_text=candidate_text,
            committed_text=committed_text,
        )
        events = tuple(self._completed_events)
        self._completed_events.clear()
        last_event = events[-1] if events else ("", "", 0.0)
        return RealtimeFrame(
            image=preview,
            sign_text=signs,
            translated_text=translation,
            confidence=confidence,
            fps=processing_fps,
            latency_ms=latency_ms,
            silent=silent,
            utterance_ended=bool(events),
            completed_text=last_event[0],
            completed_translation=last_event[1],
            camera_fps=self.camera_fps,
            completed_events=events,
            cooldown_remaining=cooldown_remaining,
            silence_remaining=silence_remaining,
            gesture_elapsed=gesture_elapsed,
            candidate_text=candidate_text,
            committed_text=committed_text,
            error=latest.error if latest else "",
        )

    def start_background(self) -> None:
        """Start the single AI consumer thread; safe to call more than once."""
        with self._worker_lock:
            if self._closed:
                raise RuntimeError("Recognizer is already closed")
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop_event.clear()
            self._worker = Thread(
                target=self._consumer_loop,
                name="sign-language-ai",
                daemon=True,
            )
            self._worker.start()

    def stop_background(self, timeout: float = 5.0) -> bool:
        """Stop the consumer and wait for its in-flight frame to finish."""
        with self._worker_lock:
            worker = self._worker
            if worker is None:
                self._collect_results()
                return True
            self._stop_event.set()
        worker.join(timeout=max(0.0, timeout))
        if worker.is_alive():
            return False
        self._collect_results()
        return True

    def drain_completed_events(self) -> tuple[tuple[str, str, float], ...]:
        self._collect_results()
        events = tuple(self._completed_events)
        self._completed_events.clear()
        return events

    def _enqueue_latest(self, job: _FrameJob) -> None:
        try:
            self.frame_queue.put_nowait(job)
            return
        except Full:
            pass
        try:
            self.frame_queue.get_nowait()
            self.frame_queue.task_done()
            self.dropped_frames += 1
        except Empty:
            pass
        try:
            self.frame_queue.put_nowait(job)
        except Full:
            self.dropped_frames += 1

    def _collect_results(self) -> None:
        with self._generation_lock:
            generation = self._generation
        while True:
            try:
                result = self.result_queue.get_nowait()
            except Empty:
                break
            try:
                if result.generation != generation:
                    continue
                self._latest_result = result
                self.confidence = result.confidence
                if result.completed_text:
                    self._completed_events.append(
                        (result.completed_text, result.completed_translation, result.confidence),
                    )
            finally:
                self.result_queue.task_done()

    def _consumer_loop(self) -> None:
        while True:
            self._apply_control_commands()
            if self._stop_event.is_set() and self.frame_queue.empty():
                break
            try:
                job = self.frame_queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                with self._generation_lock:
                    generation = self._generation
                if job.generation != generation:
                    continue
                if self._worker_generation != job.generation:
                    self._reset_worker_state()
                    self._worker_generation = job.generation
                try:
                    result = self._process_ai_frame(job)
                except Exception as exc:
                    result = self._error_result(job, exc)
                with self._generation_lock:
                    generation = self._generation
                if generation == job.generation:
                    self.result_queue.put(result)
                else:
                    self._apply_control_commands()
            finally:
                self.frame_queue.task_done()

    def _apply_control_commands(self) -> None:
        latest_generation: int | None = None
        while True:
            try:
                latest_generation = self._control_queue.get_nowait()
            except Empty:
                break
            else:
                self._control_queue.task_done()
        if latest_generation is not None and latest_generation != self._worker_generation:
            self._reset_worker_state()
            self._worker_generation = latest_generation

    def _process_ai_frame(self, job: _FrameJob) -> _InferenceResult:
        with self._state_lock:
            self.frame_count += 1
            should_process = self.previous_features is None or (self.frame_count - 1) % self.process_every_n_frames == 0
        features = self.extractor.extract(job.frame) if should_process else self.previous_features
        assert features is not None

        if self.show_landmarks:
            annotated = self.extractor.draw_landmarks(job.frame.copy())
            with self._preview_lock:
                self._landmark_preview = annotated

        with self._state_lock:
            cooldown_remaining = max(0.0, self._cooldown_until - job.captured_at)
            if cooldown_remaining > 0.0:
                self.previous_features = features.copy()
                processing_fps = self._update_fps(job.captured_at)
                return _InferenceResult(
                    generation=job.generation,
                    sign_text=self._display_sign_text,
                    translated_text=self._display_translated_text,
                    confidence=self._worker_confidence,
                    fps=processing_fps,
                    latency_ms=(time.perf_counter() - job.enqueued_at) * 1000.0,
                    silent=True,
                    cooldown_remaining=cooldown_remaining,
                    silence_remaining=0.0,
                    gesture_elapsed=0.0,
                    candidate_text=self.decode_tokens(self.pending_tokens) if self.pending_tokens else "",
                    committed_text=self.decode_tokens(self.committed_tokens) if self.committed_tokens else "",
                )

            self._update_motion(features, job.captured_at)
            self.buffer.append(features)
            silent = self.last_motion_at is not None and (job.captured_at - self.last_motion_at) >= self.silence_threshold

            # In manual_confirm mode, if user was silent and starts moving hands again, start a fresh turn score
            if self.manual_confirm and self._was_silent and not silent:
                self._turn_scores.clear()
                self._turn_started_at = job.captured_at
            elif not silent and self.last_motion_at is not None and self._turn_started_at is None:
                self._turn_started_at = job.captured_at

            if len(self.buffer) == self.sequence_length and self.frame_count - self.last_model_frame >= self.stride:
                self.last_model_frame = self.frame_count
                self._predict_window(allow_prediction=not (silent and self._was_silent), now=job.captured_at)

            completed_text = ""
            completed_translation = ""
            if not self.manual_confirm:
                if silent and not self._was_silent and self.pending_tokens:
                    completed_text = self.decode_tokens(self.pending_tokens)
                    completed_translation = self.translator.translate(self.pending_tokens)
                    self.committed_tokens.extend(self.pending_tokens)
                    self.pending_tokens.clear()
                    self.hypothesis_history.clear()
                    self._turn_scores.clear()
                    self._turn_started_at = None
                    self.buffer.clear()
                    if self.cooldown_seconds > 0.0:
                        self._cooldown_until = job.captured_at + self.cooldown_seconds
                        cooldown_remaining = self.cooldown_seconds
                elif silent and not self.pending_tokens:
                    self._turn_started_at = None
                    self._turn_scores.clear()
            else:
                if silent and not self._was_silent:
                    self._turn_started_at = None
                    self._turn_scores.clear()

            self._was_silent = silent

            silence_remaining = 0.0
            if not self.manual_confirm and self.pending_tokens and self.last_motion_at is not None and not silent:
                silence_remaining = max(0.0, self.silence_threshold - (job.captured_at - self.last_motion_at))

            gesture_elapsed = 0.0
            if self._turn_started_at is not None and not silent:
                gesture_elapsed = max(0.0, job.captured_at - self._turn_started_at)

            tokens = self.tokens
            token_key = tuple(tokens)
            if token_key != self._display_tokens:
                self._display_tokens = token_key
                self._display_sign_text = self.decode_tokens(tokens)
                self._display_translated_text = self.translator.translate(tokens)
            processing_fps = self._update_fps(job.captured_at)
            candidate_text = self.decode_tokens(self.pending_tokens) if self.pending_tokens else ""
            committed_text = self.decode_tokens(self.committed_tokens) if self.committed_tokens else ""
            return _InferenceResult(
                generation=job.generation,
                sign_text=self._display_sign_text,
                translated_text=self._display_translated_text,
                confidence=self._worker_confidence,
                fps=processing_fps,
                latency_ms=(time.perf_counter() - job.enqueued_at) * 1000.0,
                silent=silent,
                completed_text=completed_text,
                completed_translation=completed_translation,
                cooldown_remaining=cooldown_remaining,
                silence_remaining=silence_remaining,
                gesture_elapsed=gesture_elapsed,
                candidate_text=candidate_text,
                committed_text=committed_text,
            )

    def _error_result(self, job: _FrameJob, exc: Exception) -> _InferenceResult:
        return _InferenceResult(
            generation=job.generation,
            sign_text=self._display_sign_text,
            translated_text=self._display_translated_text,
            confidence=self._worker_confidence,
            fps=self.fps,
            latency_ms=(time.perf_counter() - job.enqueued_at) * 1000.0,
            silent=self._was_silent,
            error=str(exc),
        )

    def open_camera(
        self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int | None = None,
    ) -> cv2.VideoCapture:
        backends = (
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "automatic"),
        )
        for backend, _ in backends:
            capture = cv2.VideoCapture(camera_id, backend)
            if capture.isOpened():
                break
            capture.release()
        else:
            names = ", ".join(name for _, name in backends)
            raise OSError(
                f"Could not open camera {camera_id} using {names} backends. "
                "Check Windows Settings > Privacy & security > Camera and enable camera access "
                "for desktop apps, close apps that may be using the camera, and verify the "
                "camera works in the Windows Camera app."
            )
        properties = [
            (cv2.CAP_PROP_FRAME_WIDTH, width),
            (cv2.CAP_PROP_FRAME_HEIGHT, height),
            (cv2.CAP_PROP_BUFFERSIZE, 1),
        ]
        if fps is not None and fps > 0:
            properties.append((cv2.CAP_PROP_FPS, fps))
        for prop, value in properties:
            try:
                capture.set(prop, value)
            except cv2.error:
                continue
        return capture

    def reset(self) -> None:
        """Queue a reset on the consumer so model state is never mutated cross-thread."""
        with self._generation_lock:
            self._generation += 1
            generation = self._generation
        self._latest_result = None
        self._completed_events.clear()
        while True:
            try:
                self.result_queue.get_nowait()
            except Empty:
                break
            else:
                self.result_queue.task_done()
        with self._preview_lock:
            self._landmark_preview = None
        self._last_capture_at = None
        self.camera_fps = 0.0
        self.confidence = 0.0
        if self._worker is not None and self._worker.is_alive():
            self._control_queue.put(generation)
        else:
            self._reset_worker_state()
            self._worker_generation = generation

    def _reset_worker_state(self) -> None:
        with self._state_lock:
            self.buffer.clear()
            self.hypothesis_history.clear()
            self.committed_tokens.clear()
            self.pending_tokens.clear()
            self._turn_scores.clear()
            self._turn_started_at = None
            self._cooldown_until = 0.0
            self._was_silent = False
            self._worker_confidence = 0.0
            self.last_motion_at = None
            self.previous_features = None
            self.frame_count = 0
            self.last_model_frame = 0
            self.last_frame_at = None
            self.fps = 0.0
            self._display_tokens = None
            self._display_sign_text = ""
            self._display_translated_text = ""

    @property
    def tokens(self) -> list[str]:
        return [*self.committed_tokens, *self.pending_tokens]

    def decode_tokens(self, tokens: list[str]) -> str:
        token_ids = [self.tokenizer.token_to_id.get(token, self.tokenizer.unknown_id) for token in tokens]
        return self.tokenizer.decode(token_ids)

    def close(self) -> None:
        if not self.stop_background(timeout=5.0):
            raise TimeoutError("AI worker did not stop before closing the landmark extractor")
        with self._worker_lock:
            if self._closed:
                return
            self._closed = True
        self.extractor.close()

    def run(self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int | None = None) -> None:
        capture = None
        speaker = None
        latest_speech = ""
        speech_error_reported = ""
        try:
            capture = self.open_camera(camera_id, width, height, fps)
            self.start_background()
            tts_config = self.config.get("tts", {})
            if tts_config.get("enabled", False):
                from src.tts.speaker import Speaker

                speaker = Speaker(
                    rate=int(tts_config.get("rate", 165)),
                    volume=float(tts_config.get("volume", 1.0)),
                    voice_id=tts_config.get("voice_id"),
                )
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise OSError("The camera stopped returning frames")
                result = self.process_frame(frame)
                if result.error:
                    raise RuntimeError(result.error)
                for gloss, sentence, confidence in result.completed_events:
                    latest_speech = sentence
                    print(f"[ĐÃ LƯU] Từ: {gloss} | Văn bản hiện tại: {sentence} | Tin cậy: {confidence:.1%}")
                    if speaker is not None and gloss:
                        speaker.speak(gloss)
                if speaker is not None and speaker.error and speaker.error != speech_error_reported:
                    print(speaker.error)
                    speech_error_reported = speaker.error
                cv2.imshow("Sign Language AI", result.image)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key in (13, 10, ord(" ")):
                    confirmed = self.confirm_current_sign()
                    if confirmed is not None:
                        word, full_sentence, conf = confirmed
                        latest_speech = full_sentence
                        print(f"[ENTER - ĐÃ LƯU] +{word} => Câu văn: {full_sentence} (Tin cậy: {conf:.1%})")
                        if speaker is not None and word:
                            speaker.speak(word)
                elif key == 8:
                    updated_sentence = self.pop_committed_sign()
                    latest_speech = updated_sentence
                    print(f"[BACKSPACE - XÓA 1 TỪ] Câu văn hiện tại: {updated_sentence or '(trống)'}")
                elif key == ord("r"):
                    self.reset()
                    print("[RESET] Đã xóa toàn bộ văn bản.")
                elif key == ord("l"):
                    self.show_landmarks = not self.show_landmarks
                elif key == ord("s") and speaker is not None:
                    with self._state_lock:
                        sentence_to_speak = (
                            self.translator.translate(self.committed_tokens)
                            if self.committed_tokens
                            else (self.translator.translate(self.pending_tokens) if self.pending_tokens else latest_speech)
                        )
                    if sentence_to_speak:
                        print(f"[ĐỌC CÂU VĂN]: {sentence_to_speak}")
                        speaker.speak(sentence_to_speak)
        finally:
            if capture is not None:
                capture.release()
            cv2.destroyAllWindows()
            if speaker is not None:
                speaker.close()
            self.close()

    def _predict_window(self, *, allow_prediction: bool = True, now: float | None = None) -> None:
        if not allow_prediction:
            return
        sequence = np.stack(self.buffer).astype(np.float32, copy=False)
        if self.require_hand and self.feature_config.use_hands:
            if sequence.shape[1] == 201:
                has_lh = (sequence[:, 75:138] != 0.0).any(axis=1)
                has_rh = (sequence[:, 138:201] != 0.0).any(axis=1)
            elif sequence.shape[1] % 4 == 0:
                has_lh = (sequence[:, 3:84:4] > 0.5).any(axis=1)
                has_rh = (sequence[:, 87:168:4] > 0.5).any(axis=1)
            else:
                has_lh = np.ones(sequence.shape[0], dtype=bool)
                has_rh = np.ones(sequence.shape[0], dtype=bool)
            hand_mask = has_lh | has_rh
            hand_frames = int(hand_mask.sum())
            min_hand_frames = max(1, int(self.sequence_length * 0.25))
            if hand_frames < min_hand_frames:
                self._worker_confidence = 0.0
                self.hypothesis_history.clear()
                return
        input_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        lengths = torch.tensor([sequence.shape[0]], dtype=torch.long)
        with torch.inference_mode():
            log_probs = self.model(input_tensor, lengths)
        decoded = self.decoder.decode(log_probs, lengths)[0]
        best_ids = log_probs.detach().argmax(dim=-1).squeeze(1).tolist()
        probs = log_probs.detach().float().exp().squeeze(1)
        non_blank = [int(idx) for idx in best_ids if idx != self.tokenizer.blank_id and idx != self.tokenizer.unknown_id]
        confidence = decoded.confidence
        if non_blank:
            counts = Counter(non_blank)
            dominant_id = max(
                counts,
                key=lambda cid: (counts[cid], float(probs[:, cid].mean().item())),
            )
            raw_tokens = [self.tokenizer.tokens[dominant_id]]
            peak_conf = float(probs[:, dominant_id].max().item())
            top_k = max(1, min(5, probs.shape[0]))
            topk_conf = float(torch.topk(probs[:, dominant_id], k=top_k).values.mean().item())
            confidence = max(confidence, 0.6 * peak_conf + 0.4 * topk_conf)
        elif decoded.token_ids:
            meaningful_ids = [idx for idx in decoded.token_ids if idx != self.tokenizer.unknown_id]
            if meaningful_ids:
                dominant_id = meaningful_ids[-1]
                raw_tokens = [self.tokenizer.tokens[dominant_id]]
                top_k = max(1, min(5, probs.shape[0]))
                topk_conf = float(torch.topk(probs[:, dominant_id], k=top_k).values.mean().item())
                peak_conf = float(probs[:, dominant_id].max().item())
                confidence = max(confidence, 0.5 * peak_conf + 0.5 * topk_conf)
            else:
                raw_tokens = []
        else:
            raw_tokens = []
        filtered = confidence_filter(raw_tokens, confidence, self.confidence_threshold)
        self._worker_confidence = confidence
        self.hypothesis_history.append(filtered)
        stable = temporal_smoothing(
            list(self.hypothesis_history),
            min_votes=self.smoothing_votes,
            window_size=self.smoothing_window,
        )
        if stable:
            candidate = stable[-1]
            if self._turn_started_at is None and now is not None:
                self._turn_started_at = now
            if self.single_word_per_turn:
                for existing_tok in list(self._turn_scores):
                    self._turn_scores[existing_tok] *= 0.85
                self._turn_scores[candidate] += confidence
                best_turn_token = max(self._turn_scores, key=lambda tok: self._turn_scores[tok])
                self.pending_tokens = [best_turn_token]
            else:
                if not self.pending_tokens or self.pending_tokens[-1] != candidate:
                    self.pending_tokens.append(candidate)

    def _update_motion(self, features: np.ndarray, current_time: float) -> None:
        if self.previous_features is not None:
            if features.shape[0] % 4 == 0:
                previous = self.previous_features.reshape(-1, 4)
                current = features.reshape(-1, 4)
                visible = (previous[:, 3] > 0.5) & (current[:, 3] > 0.5)
                if visible.any():
                    motion = float(np.linalg.norm(current[visible, :3] - previous[visible, :3], axis=1).mean())
                else:
                    motion = 0.0
            elif features.shape[0] % 3 == 0:
                previous = self.previous_features.reshape(-1, 3)
                current = features.reshape(-1, 3)
                visible = (previous != 0.0).any(axis=-1) & (current != 0.0).any(axis=-1)
                if visible.any():
                    motion = float(np.linalg.norm(current[visible] - previous[visible], axis=1).mean())
                else:
                    motion = 0.0
            else:
                motion = float(np.linalg.norm(features - self.previous_features))

            if motion >= self.motion_threshold:
                self.last_motion_at = current_time
        self.previous_features = features.copy()

    def _update_fps(self, now: float) -> float:
        if self.last_frame_at is not None:
            elapsed = now - self.last_frame_at
            if elapsed > 0:
                instantaneous = 1.0 / elapsed
                self.fps = instantaneous if self.fps == 0 else 0.85 * self.fps + 0.15 * instantaneous
        self.last_frame_at = now
        return self.fps

    def _update_capture_fps(self, now: float) -> float:
        if self._last_capture_at is not None:
            elapsed = now - self._last_capture_at
            if elapsed > 0:
                instantaneous = 1.0 / elapsed
                self.camera_fps = instantaneous if self.camera_fps == 0 else 0.85 * self.camera_fps + 0.15 * instantaneous
        self._last_capture_at = now
        return self.camera_fps

    def _format_timer_status(
        self,
        *,
        silent: bool,
        cooldown_remaining: float,
        silence_remaining: float,
        gesture_elapsed: float,
        candidate_text: str = "",
    ) -> str:
        if cooldown_remaining > 0.0:
            return f"Đã lưu từ! Nghỉ hạ tay: còn {cooldown_remaining:.1f}s"
        if self.manual_confirm:
            if candidate_text:
                if not silent and gesture_elapsed > 0.0:
                    return f"Đang làm động tác ({gesture_elapsed:.1f}s) -> Nhấn ENTER để lưu '{candidate_text}'"
                return f"Đã bắt được từ '{candidate_text}' -> Nhấn ENTER để lưu (hoặc làm lại để đổi)"
            if gesture_elapsed > 0.0 and not silent:
                return f"Đang phân tích động tác: {gesture_elapsed:.1f}s..."
            return "Sẵn sàng (0.0s) - Hãy làm 1 động tác rồi nhấn ENTER để lưu vào câu"
        if self.pending_tokens and silence_remaining > 0.0:
            return f"Thời gian lượt: {gesture_elapsed:.1f}s  |  Giữ yên/hạ tay để chốt: còn {silence_remaining:.1f}s"
        if gesture_elapsed > 0.0 and not silent:
            return f"Đang thực hiện động tác: {gesture_elapsed:.1f}s  (Dừng tay {self.silence_threshold:.1f}s để chốt)"
        if silent:
            return "Đã kết thúc lượt (0.0s)  |  Sẵn sàng nhận ký hiệu tiếp theo"
        return f"Sẵn sàng (0.0s)  |  Mỗi lượt giữ/dừng tay {self.silence_threshold:.1f}s để chốt từ"

    def _draw_status(
        self,
        frame: np.ndarray,
        signs: str,
        translation: str,
        confidence: float,
        processing_fps: float,
        latency_ms: float,
        silent: bool,
        *,
        cooldown_remaining: float = 0.0,
        silence_remaining: float = 0.0,
        gesture_elapsed: float = 0.0,
        candidate_text: str = "",
        committed_text: str = "",
    ) -> None:
        timer_line = self._format_timer_status(
            silent=silent,
            cooldown_remaining=cooldown_remaining,
            silence_remaining=silence_remaining,
            gesture_elapsed=gesture_elapsed,
            candidate_text=candidate_text,
        )
        if self.manual_confirm:
            rows = [
                "Hỗ trợ giao tiếp bằng ngôn ngữ ký hiệu (Chế độ nhấn ENTER để lưu)",
                f"Từ đang nhận diện (1 từ): {candidate_text or '...'}",
                f"Văn bản đã lưu: {translation or committed_text or '(Chưa có - Nhấn Enter khi thấy từ đúng)'}",
                f"Hướng dẫn & Nhịp: {timer_line}",
                f"Tin cậy: {confidence:.1%}  AI: {processing_fps:.1f} FPS  Camera: {self.camera_fps:.1f} FPS  Trễ: {latency_ms:.1f} ms",
                "Enter/Space: Lưu từ   Backspace: Xóa 1 từ   R: Xóa hết   S: Đọc câu   L: Khung xương   Q: Thoát",
            ]
        else:
            rows = [
                "Hỗ trợ giao tiếp bằng ngôn ngữ ký hiệu",
                f"Ký hiệu nhận diện: {signs or '...'}",
                f"Câu tiếng Việt (thử nghiệm): {translation or '...'}",
                f"Bộ đếm nhịp: {timer_line}",
                f"Tin cậy: {confidence:.1%}  AI: {processing_fps:.1f} FPS  Camera: {self.camera_fps:.1f} FPS  Trễ: {latency_ms:.1f} ms",
                f"Thiết bị: {self.device_label}" + ("  Đã kết thúc lượt ký hiệu" if silent else ""),
                "R: Xóa   L: Khung xương   S: Đọc lại   Q/Esc: Thoát"
                if self.config.get("tts", {}).get("enabled", False)
                else "R: Xóa   L: Khung xương   Q/Esc: Thoát",
            ]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).convert("RGBA")
        try:
            font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        padding = 12
        max_width = max(1, image.width - padding * 2)
        draw_probe = ImageDraw.Draw(image)
        lines: list[tuple[str, tuple[int, int, int, int]]] = []
        for row_idx, row in enumerate(rows):
            if self.manual_confirm and row_idx == 1 and candidate_text:
                color = (255, 230, 80, 255)  # Yellow highlight for active candidate word
            elif self.manual_confirm and row_idx == 2 and (translation or committed_text):
                color = (100, 255, 160, 255)  # Green highlight for saved sentence
            elif row_idx == 3:
                color = (130, 235, 255, 255)  # Cyan for timer/guide line
            else:
                color = (255, 255, 255, 255)
            words = row.split()
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if current and draw_probe.textbbox((0, 0), candidate, font=font)[2] > max_width:
                    lines.append((current, color))
                    current = word
                else:
                    current = candidate
            if current:
                lines.append((current, color))
        line_height = 25
        panel_height = min(image.height, padding * 2 + line_height * len(lines))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle((0, 0, image.width, panel_height), fill=(0, 0, 0, 190))
        image = Image.alpha_composite(image, overlay)
        draw = ImageDraw.Draw(image)
        for index, (text, fill_color) in enumerate(lines):
            y = padding + index * line_height
            if y + line_height > image.height:
                break
            draw.text((padding, y), text, font=font, fill=fill_color)
        frame[:] = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
