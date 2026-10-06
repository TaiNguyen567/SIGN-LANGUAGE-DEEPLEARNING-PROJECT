from types import SimpleNamespace

import numpy as np

from src.features.landmark_extractor import FeatureConfig, LandmarkExtractor


class ProtoLandmark:
    def __init__(self, x, y, z=0.0, *, visibility=None, presence=None):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility or 0.0
        self.presence = presence or 0.0
        self._set_fields = {
            name for name, value in (("visibility", visibility), ("presence", presence)) if value is not None
        }

    def HasField(self, name):
        return name in self._set_fields


def test_missing_landmarks_produce_zero_values_and_masks():
    extractor = LandmarkExtractor(detector=object())
    empty_result = SimpleNamespace(
        left_hand_landmarks=None,
        right_hand_landmarks=None,
        pose_landmarks=None,
        face_landmarks=None,
    )

    features = extractor.create_feature_vector(empty_result)

    assert features.shape == (extractor.feature_dim,)
    assert features.dtype == np.float32
    assert np.count_nonzero(features) == 0


def test_visibility_is_used_when_presence_is_unset_and_hands_need_no_confidence_fields():
    hands = SimpleNamespace(landmark=[ProtoLandmark(i / 20, i / 40) for i in range(21)])
    pose = SimpleNamespace(landmark=[
        ProtoLandmark(i / 32, i / 64, visibility=0.9) for i in range(33)
    ])
    result = SimpleNamespace(
        left_hand_landmarks=hands,
        right_hand_landmarks=None,
        pose_landmarks=pose,
        face_landmarks=None,
    )
    extractor = LandmarkExtractor(detector=object())

    features = extractor.create_feature_vector(result).reshape(-1, 4)

    assert features[:21, 3].sum() == 21
    assert features[21:42, 3].sum() == 0
    assert features[42:, 3].sum() == 33
    assert np.count_nonzero(features[:, :3]) > 0


def test_explicit_low_visibility_landmarks_are_still_filtered():
    pose_points = [ProtoLandmark(i / 32, i / 64, visibility=0.9) for i in range(33)]
    pose_points[0] = ProtoLandmark(0.1, 0.1, visibility=0.1)
    result = SimpleNamespace(
        left_hand_landmarks=None,
        right_hand_landmarks=None,
        pose_landmarks=SimpleNamespace(landmark=pose_points),
        face_landmarks=None,
    )
    extractor = LandmarkExtractor(FeatureConfig(use_hands=False), detector=object())

    features = extractor.create_feature_vector(result).reshape(-1, 4)

    assert features[:, 3].sum() == 32


def test_normalization_is_translation_and_scale_invariant_for_hands():
    extractor = LandmarkExtractor(detector=object())
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.linspace(0.1, 0.2, 21)
    points[:, 1] = np.linspace(0.3, 0.4, 21)
    points[9, :2] = (0.3, 0.3)

    first = extractor.normalize_landmarks(
        points, kind="hand", expected_count=21, valid_mask=np.ones(21, dtype=bool),
        reference_indices=(0,), scale_indices=(0, 9),
    )
    transformed = points * 2.0 + 0.15
    second = extractor.normalize_landmarks(
        transformed, kind="hand", expected_count=21, valid_mask=np.ones(21, dtype=bool),
        reference_indices=(0,), scale_indices=(0, 9),
    )

    np.testing.assert_allclose(first, second, atol=1e-5)


def test_disabled_landmark_groups_reduce_feature_dimension():
    extractor = LandmarkExtractor(
        FeatureConfig(use_hands=False, use_pose=True, use_face=False), detector=object(),
    )
    assert extractor.feature_dim == 4 * 33
