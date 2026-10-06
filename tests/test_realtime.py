import cv2
import numpy as np
import pytest
import torch
from threading import Event, get_ident

from src.features.landmark_extractor import FeatureConfig
from src.inference.realtime import RealtimeRecognizer
from src.data.tokenizer import SignTokenizer


class _FakeExtractor:
    def extract(self, frame):
        return np.zeros(FeatureConfig().feature_dim, dtype=np.float32)

    def draw_landmarks(self, frame):
        return frame

    def close(self):
        return None


class _BlankModel:
    def __init__(self):
        self.calls = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, features, lengths):
        self.calls += 1
        logits = torch.zeros(features.shape[1], 1, 3, device=features.device)
        logits[:, :, 0] = 8.0
        return logits.log_softmax(-1)


class _TokenModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, features, lengths):
        logits = torch.full((features.shape[1], 1, 3), -8.0, device=features.device)
        logits[:, :, 2] = 8.0
        return logits.log_softmax(-1)


class _NoisyWordModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, features, lengths):
        path = torch.tensor([2, 2, 2, 3], device=features.device).view(4, 1, 1)
        logits = torch.full((4, 1, 4), -8.0, device=features.device)
        logits.scatter_(2, path, 8.0)
        return logits.log_softmax(-1)


class _MovingExtractor(_FakeExtractor):
    def __init__(self):
        self.index = 0
        self.coordinate = 0.0

    def extract(self, frame):
        values = np.zeros((75, 4), dtype=np.float32)
        values[:, 3] = 1.0
        if self.index < 6:
            self.coordinate += 0.1
        values[:, 0] = self.coordinate
        self.index += 1
        return values.reshape(-1)


class _BlockingExtractor(_FakeExtractor):
    def __init__(self):
        self.entered = Event()
        self.release = Event()
        self.frame_values = []
        self.thread_ids = []

    def extract(self, frame):
        self.frame_values.append(int(frame[0, 0, 0]))
        self.thread_ids.append(get_ident())
        self.entered.set()
        self.release.wait(timeout=2.0)
        return np.zeros(FeatureConfig().feature_dim, dtype=np.float32)


@pytest.fixture
def recognizer_factory():
    created = []

    def build(*args, **kwargs):
        recognizer = RealtimeRecognizer(*args, **kwargs)
        created.append(recognizer)
        return recognizer

    yield build
    for recognizer in created:
        recognizer.close()


def _submit_and_wait(recognizer, frame, now):
    display_result = recognizer.process_frame(frame, now=now)
    recognizer.frame_queue.join()
    recognizer._collect_results()
    return display_result, recognizer._latest_result


def test_realtime_uses_overlapping_sequence_windows_without_camera_capture(recognizer_factory):
    model = _BlankModel()
    config = {
        "data": {"sequence_length": 4, "stride": 2},
        "realtime": {"show_landmarks": False, "confidence_threshold": 0.1},
    }
    recognizer = recognizer_factory(
        model, SignTokenizer.build(["TÔI"]), device=torch.device("cpu"),
        feature_config=FeatureConfig(), config=config, extractor=_FakeExtractor(),
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    for index in range(6):
        display, state = _submit_and_wait(recognizer, frame, now=100.0 + index / 30.0)

    assert model.calls == 2
    assert display.image.shape == frame.shape
    assert state.sign_text == ""


def test_realtime_prediction_stride_overrides_training_stride(recognizer_factory):
    config = {"data": {"sequence_length": 4, "stride": 8}, "realtime": {"prediction_stride": 2}}
    recognizer = recognizer_factory(
        _BlankModel(), SignTokenizer.build(["TÔI"]), device=torch.device("cpu"),
        feature_config=FeatureConfig(), config=config, extractor=_FakeExtractor(),
    )

    assert recognizer.stride == 2


def test_camera_overlay_renders_vietnamese_text(recognizer_factory):
    recognizer = recognizer_factory(
        _BlankModel(), SignTokenizer.build(["CẢM ƠN"]), device=torch.device("cpu"),
        feature_config=FeatureConfig(), config={"realtime": {"show_landmarks": False}},
        extractor=_FakeExtractor(),
    )
    frame = np.zeros((240, 640, 3), dtype=np.uint8)

    recognizer._draw_status(frame, "CẢM ƠN", "Cảm ơn.", 0.9, 24.0, 20.0, False)

    assert frame.shape == (240, 640, 3)
    assert frame.sum() > 0


def test_realtime_word_checkpoint_keeps_one_stable_class_per_window(recognizer_factory):
    tokenizer = SignTokenizer(["<blank>", "<unk>", "TÔI", "BẠN"])
    config = {
        "data": {"sequence_length": 4, "stride": 8},
        "realtime": {"prediction_stride": 1, "confidence_threshold": 0.1},
    }
    recognizer = recognizer_factory(
        _NoisyWordModel(), tokenizer, device=torch.device("cpu"),
        feature_config=FeatureConfig(), config=config, extractor=_FakeExtractor(),
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    for index in range(5):
        _, result = _submit_and_wait(recognizer, frame, now=100.0 + index / 30.0)

    assert result.sign_text == ""
    assert recognizer.pending_tokens == []

    _, result = _submit_and_wait(recognizer, frame, now=100.0 + 5 / 30.0)
    assert result.sign_text == "TÔI"
    assert recognizer.pending_tokens == ["TÔI"]


def test_producer_stays_nonblocking_and_queue_keeps_only_latest_frame(recognizer_factory):
    extractor = _BlockingExtractor()
    recognizer = recognizer_factory(
        _BlankModel(), SignTokenizer.build(["TÔI"]), device=torch.device("cpu"),
        feature_config=FeatureConfig(),
        config={"data": {"sequence_length": 4}, "realtime": {"show_landmarks": False}},
        extractor=extractor,
    )
    producer_thread = get_ident()

    def camera_frame(value):
        return np.full((8, 8, 3), value, dtype=np.uint8)

    recognizer.process_frame(camera_frame(1), now=1.0)
    assert extractor.entered.wait(timeout=1.0)
    recognizer.process_frame(camera_frame(2), now=1.03)
    recognizer.process_frame(camera_frame(3), now=1.06)
    assert recognizer.frame_queue.qsize() == 1
    assert recognizer.dropped_frames == 1

    extractor.release.set()
    assert recognizer.stop_background(timeout=2.0)
    assert extractor.frame_values == [1, 3]
    assert all(thread_id != producer_thread for thread_id in extractor.thread_ids)


def test_reset_is_applied_on_the_consumer_thread(recognizer_factory):
    recognizer = recognizer_factory(
        _NoisyWordModel(), SignTokenizer(["<blank>", "<unk>", "TÔI", "BẠN"]),
        device=torch.device("cpu"), feature_config=FeatureConfig(),
        config={
            "data": {"sequence_length": 4},
            "realtime": {"prediction_stride": 1, "confidence_threshold": 0.1, "show_landmarks": False},
        },
        extractor=_FakeExtractor(),
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    for index in range(6):
        _submit_and_wait(recognizer, frame, now=100.0 + index / 30.0)
    assert recognizer.pending_tokens == ["TÔI"]

    recognizer.reset()
    for index in range(4):
        _submit_and_wait(recognizer, frame, now=101.0 + index / 30.0)

    assert recognizer.pending_tokens == []
    assert recognizer.committed_tokens == []


def test_realtime_rejects_predictions_below_confidence_threshold(recognizer_factory):
    tokenizer = SignTokenizer(["<blank>", "<unk>", "TÔI", "BẠN"])
    config = {
        "data": {"sequence_length": 4, "stride": 8},
        "realtime": {"prediction_stride": 1, "confidence_threshold": 1.0},
    }
    recognizer = recognizer_factory(
        _NoisyWordModel(), tokenizer, device=torch.device("cpu"),
        feature_config=FeatureConfig(), config=config, extractor=_FakeExtractor(),
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    for index in range(8):
        _, result = _submit_and_wait(recognizer, frame, now=100.0 + index / 30.0)

    assert result.sign_text == ""
    assert recognizer.pending_tokens == []


def test_silence_commits_an_utterance_without_a_user_action(recognizer_factory):
    config = {
        "data": {"sequence_length": 4, "stride": 1},
        "realtime": {
            "show_landmarks": False, "confidence_threshold": 0.1,
            "silence_threshold": 0.3, "motion_threshold": 0.01,
        },
    }
    tokenizer = SignTokenizer(["<blank>", "<unk>", "HELLO"])
    recognizer = recognizer_factory(
        _TokenModel(), tokenizer, device=torch.device("cpu"),
        feature_config=FeatureConfig(), config=config, extractor=_MovingExtractor(),
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    results = [_submit_and_wait(recognizer, frame, now=i / 10)[1] for i in range(9)]
    completed = recognizer.drain_completed_events()

    assert len(completed) == 1
    assert completed[0][:2] == ("HELLO", "Hello.")
    assert results[-1].sign_text == "HELLO"


def test_realtime_reuses_translation_while_tokens_are_unchanged(recognizer_factory):
    class CountingTranslator:
        def __init__(self):
            self.calls = 0

        def translate(self, tokens):
            self.calls += 1
            return ""

    config = {
        "data": {"sequence_length": 4, "stride": 2},
        "realtime": {"show_landmarks": False},
    }
    recognizer = recognizer_factory(
        _BlankModel(), SignTokenizer.build(["TÔI"]), device=torch.device("cpu"),
        feature_config=FeatureConfig(), config=config, extractor=_FakeExtractor(),
    )
    translator = CountingTranslator()
    recognizer.translator = translator
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    for index in range(6):
        _submit_and_wait(recognizer, frame, now=index / 30.0)

    assert translator.calls == 1


def _recognizer_for_camera_test(recognizer_factory):
    return recognizer_factory(
        _BlankModel(), SignTokenizer.build(["TÔI"]), device=torch.device("cpu"),
        feature_config=FeatureConfig(), config={}, extractor=_FakeExtractor(),
    )


def test_open_camera_falls_back_to_media_foundation(monkeypatch, recognizer_factory):
    class FakeCapture:
        def __init__(self, opened):
            self.opened = opened
            self.released = False
            self.properties = []

        def isOpened(self):
            return self.opened

        def release(self):
            self.released = True

        def set(self, prop, value):
            self.properties.append((prop, value))
            return True

    attempts = []
    captures = [FakeCapture(False), FakeCapture(True)]

    def video_capture(camera_id, backend):
        attempts.append((camera_id, backend))
        return captures[len(attempts) - 1]

    monkeypatch.setattr("src.inference.realtime.cv2.VideoCapture", video_capture)
    recognizer = _recognizer_for_camera_test(recognizer_factory)

    capture = recognizer.open_camera(1, 640, 480, 24)

    assert attempts == [(1, cv2.CAP_DSHOW), (1, cv2.CAP_MSMF)]
    assert captures[0].released
    assert not captures[1].released
    assert captures[1].properties == [
        (cv2.CAP_PROP_FRAME_WIDTH, 640),
        (cv2.CAP_PROP_FRAME_HEIGHT, 480),
        (cv2.CAP_PROP_BUFFERSIZE, 1),
        (cv2.CAP_PROP_FPS, 24),
    ]


def test_open_camera_ignores_unsupported_capture_properties(monkeypatch, recognizer_factory):
    class FakeCapture:
        def __init__(self):
            self.properties = []

        def isOpened(self):
            return True

        def release(self):
            pass

        def set(self, prop, value):
            self.properties.append((prop, value))
            if prop == cv2.CAP_PROP_FPS:
                raise cv2.error("FPS property is unsupported")
            return True

    capture = FakeCapture()
    monkeypatch.setattr("src.inference.realtime.cv2.VideoCapture", lambda *_: capture)
    recognizer = _recognizer_for_camera_test(recognizer_factory)

    assert recognizer.open_camera(0, 640, 480, 60) is capture
    assert capture.properties[-1] == (cv2.CAP_PROP_FPS, 60)


def test_open_camera_reports_windows_access_checks_when_all_backends_fail(monkeypatch, recognizer_factory):
    class FakeCapture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    captures = []

    def video_capture(camera_id, backend):
        capture = FakeCapture()
        captures.append((camera_id, backend, capture))
        return capture

    monkeypatch.setattr("src.inference.realtime.cv2.VideoCapture", video_capture)
    recognizer = _recognizer_for_camera_test(recognizer_factory)

    with pytest.raises(OSError, match="Privacy & security > Camera"):
        recognizer.open_camera(2)

    assert [backend for _, backend, _ in captures] == [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    assert all(capture.released for _, _, capture in captures)
