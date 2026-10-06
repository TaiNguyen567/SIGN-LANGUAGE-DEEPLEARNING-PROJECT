"""MediaPipe Holistic adapter that produces normalized frame feature vectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.features.normalization import LandmarkNormalizer


DEFAULT_FACE_INDICES = (
    1, 10, 33, 61, 70, 105, 133, 152, 159, 234, 263, 291, 300, 334, 362, 386, 454,
)


@dataclass(frozen=True)
class FeatureConfig:
    """Select landmark groups while keeping a deterministic feature layout."""

    use_hands: bool = True
    use_pose: bool = True
    use_face: bool = False
    face_landmark_indices: tuple[int, ...] = field(default_factory=lambda: DEFAULT_FACE_INDICES)
    min_visibility: float = 0.5
    model_complexity: int = 1

    @property
    def feature_dim(self) -> int:
        point_count = 0
        if self.use_hands:
            point_count += 2 * 21
        if self.use_pose:
            point_count += 33
        if self.use_face:
            point_count += len(self.face_landmark_indices)
        return point_count * 4


class LandmarkExtractor:
    """Extract BGR video frames into normalized hand, pose, and face features.

    Each enabled landmark is represented by ``[x, y, z, valid]``. Coordinates
    are centered and scaled per body part; missing landmarks remain all-zero.
    """

    HAND_COUNT = 21
    POSE_COUNT = 33
    FACE_COUNT = len(DEFAULT_FACE_INDICES)

    def __init__(
        self,
        config: FeatureConfig | None = None,
        *,
        detector: Any | None = None,
        normalizer: LandmarkNormalizer | None = None,
    ) -> None:
        self.config = config or FeatureConfig()
        self.normalizer = normalizer or LandmarkNormalizer()
        self._owns_detector = detector is None
        self.detector = detector if detector is not None else self._create_detector()
        self.last_result: Any | None = None

    @property
    def feature_dim(self) -> int:
        """Number of floats emitted for each frame."""
        return self.config.feature_dim

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Run MediaPipe on one OpenCV BGR frame and return a flat vector."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape [height, width, 3]")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to process image frames") from exc

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb_frame)
        self.last_result = result
        return self.create_feature_vector(result)

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw available MediaPipe landmarks on a BGR frame for local preview."""
        if self.last_result is None:
            return frame
        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            return frame
        result = self.last_result
        if self.config.use_pose and result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
            )
        if self.config.use_hands:
            for points in (result.left_hand_landmarks, result.right_hand_landmarks):
                if points:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, points, mp.solutions.holistic.HAND_CONNECTIONS,
                    )
        if self.config.use_face and result.face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
            )
        return frame

    def extract_sequence(self, video: str | Path) -> np.ndarray:
        """Read a video incrementally and return its feature sequence [T, F]."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to read video files") from exc

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Could not open video: {video}")

        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(self.extract(frame))
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"Video contains no readable frames: {video}")
        return np.stack(frames).astype(np.float32, copy=False)

    def normalize_landmarks(
        self,
        landmarks: np.ndarray | None,
        *,
        kind: str,
        expected_count: int,
        valid_mask: np.ndarray | None = None,
        reference_indices: tuple[int, ...] = (),
        scale_indices: tuple[int, ...] = (),
    ) -> np.ndarray:
        """Normalize one landmark group; exposed for dataset preparation/tests."""
        return self.normalizer.normalize(
            landmarks,
            kind=kind,
            expected_count=expected_count,
            valid_mask=valid_mask,
            reference_indices=reference_indices,
            scale_indices=scale_indices,
        )

    def create_feature_vector(self, result: Any) -> np.ndarray:
        """Convert a MediaPipe Holistic result into the configured feature layout."""
        groups: list[np.ndarray] = []
        if self.config.use_hands:
            groups.extend((
                self._normalize_result_group(getattr(result, "left_hand_landmarks", None), "hand", self.HAND_COUNT,
                                             reference_indices=(0,), scale_indices=(0, 9)),
                self._normalize_result_group(getattr(result, "right_hand_landmarks", None), "hand", self.HAND_COUNT,
                                             reference_indices=(0,), scale_indices=(0, 9)),
            ))
        if self.config.use_pose:
            groups.append(self._normalize_result_group(
                getattr(result, "pose_landmarks", None), "pose", self.POSE_COUNT,
                reference_indices=(11, 12), scale_indices=(11, 12),
            ))
        if self.config.use_face:
            selected = self._select_face_landmarks(getattr(result, "face_landmarks", None))
            face_positions = tuple(i for i, source_i in enumerate(self.config.face_landmark_indices) if source_i == 1)
            eye_positions = tuple(i for i, source_i in enumerate(self.config.face_landmark_indices) if source_i in (33, 263))
            groups.append(self._normalize_array(
                selected[0], selected[1], "face", len(self.config.face_landmark_indices),
                reference_indices=face_positions, scale_indices=eye_positions,
            ))
        if not groups:
            raise ValueError("At least one of hands, pose, or face must be enabled")
        return np.concatenate([group.reshape(-1) for group in groups]).astype(np.float32, copy=False)

    def close(self) -> None:
        """Release MediaPipe resources owned by this extractor."""
        if self._owns_detector:
            close = getattr(self.detector, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _normalize_result_group(
        self,
        group: Any,
        kind: str,
        count: int,
        *,
        reference_indices: tuple[int, ...],
        scale_indices: tuple[int, ...],
    ) -> np.ndarray:
        points = self._landmark_list(group)
        if points is None:
            return self.normalize_landmarks(None, kind=kind, expected_count=count)
        coords, valid = self._points_to_arrays(points, count)
        return self._normalize_array(
            coords, valid, kind, count,
            reference_indices=reference_indices, scale_indices=scale_indices,
        )

    def _normalize_array(
        self,
        coords: np.ndarray,
        valid: np.ndarray,
        kind: str,
        count: int,
        *,
        reference_indices: tuple[int, ...],
        scale_indices: tuple[int, ...],
    ) -> np.ndarray:
        return self.normalize_landmarks(
            coords, kind=kind, expected_count=count, valid_mask=valid,
            reference_indices=reference_indices, scale_indices=scale_indices,
        )

    def _group_specs(self) -> list[tuple[str, int, Any]]:
        groups: list[tuple[str, int, Any]] = []
        if self.config.use_hands:
            groups.extend((("left_hand", self.HAND_COUNT, None), ("right_hand", self.HAND_COUNT, None)))
        if self.config.use_pose:
            groups.append(("pose", self.POSE_COUNT, None))
        if self.config.use_face:
            groups.append(("face", len(self.config.face_landmark_indices), None))
        return groups

    def _select_face_landmarks(self, group: Any) -> tuple[np.ndarray, np.ndarray]:
        points = self._landmark_list(group)
        return self._points_to_arrays(points, len(self.config.face_landmark_indices), self.config.face_landmark_indices)

    def _points_to_arrays(
        self,
        points: list[Any] | None,
        count: int,
        indices: tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        coords = np.zeros((count, 3), dtype=np.float32)
        valid = np.zeros(count, dtype=bool)
        if points is None:
            return coords, valid
        for output_i in range(count):
            source_i = indices[output_i] if indices is not None and output_i < len(indices) else output_i
            if source_i >= len(points):
                continue
            point = points[source_i]
            xyz = np.asarray((getattr(point, "x", 0.0), getattr(point, "y", 0.0), getattr(point, "z", 0.0)), dtype=np.float32)
            confidence_ok = True
            has_field = getattr(point, "HasField", None)
            for confidence_name in ("visibility", "presence"):
                if not hasattr(point, confidence_name):
                    continue
                if callable(has_field):
                    try:
                        if not has_field(confidence_name):
                            continue
                    except (TypeError, ValueError):
                        pass
                confidence = float(getattr(point, confidence_name))
                if not np.isfinite(confidence) or confidence < self.config.min_visibility:
                    confidence_ok = False
                    break
            is_valid = np.isfinite(xyz).all() and confidence_ok
            if is_valid:
                coords[output_i] = xyz
                valid[output_i] = True
        return coords, valid

    @staticmethod
    def _landmark_list(group: Any) -> list[Any] | None:
        if group is None:
            return None
        points = getattr(group, "landmark", group)
        try:
            return list(points)
        except TypeError:
            return None

    def _create_detector(self) -> Any:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is required; install the project requirements first") from exc
        return mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
