import numpy as np

from src.data.augmentation import LandmarkAugmenter


def test_landmark_augmentation_preserves_validity_layout_and_finite_values():
    sequence = np.zeros((12, 8), dtype=np.float32)
    sequence[:, 0] = np.linspace(-0.2, 0.2, 12)
    sequence[:, 1] = 0.1
    sequence[:, 3] = 1.0
    sequence[:, 4:7] = 0.25
    sequence[:, 7] = 1.0
    augmenter = LandmarkAugmenter({
        "enabled": True,
        "gaussian_noise_std": 0.01,
        "scale_range": [0.95, 1.05],
        "translation_std": 0.01,
        "rotation_degrees": 5,
        "temporal_crop_probability": 1.0,
        "temporal_stretch_probability": 1.0,
        "frame_drop_probability": 0.05,
        "landmark_mask_probability": 0.2,
    }, seed=4)

    augmented = augmenter(sequence)

    assert augmented.ndim == 2 and augmented.shape[1] == 8
    assert augmented.shape[0] >= 2
    assert np.isfinite(augmented).all()
    assert set(np.unique(augmented[:, 3::4])).issubset({0.0, 1.0})


def test_disabled_augmentation_returns_an_equal_copy():
    sequence = np.ones((3, 8), dtype=np.float32)
    output = LandmarkAugmenter({"enabled": False})(sequence)
    np.testing.assert_array_equal(output, sequence)
    assert output is not sequence


def test_temporal_stretch_matches_linear_interpolation_for_all_coordinates():
    class FixedRng:
        def random(self):
            return 0.5

        def uniform(self, low, high):
            return high

    sequence = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    sequence[:, 3] = np.array([1, 0, 1, 1, 0, 1], dtype=np.float32)
    sequence[:, 7] = np.array([0, 1, 1, 0, 1, 1], dtype=np.float32)
    augmenter = LandmarkAugmenter({
        "enabled": True,
        "temporal_crop_probability": 0.0,
        "temporal_stretch_probability": 1.0,
        "frame_drop_probability": 0.0,
    })
    augmenter.rng = FixedRng()

    actual = augmenter._temporal_augment(sequence)
    output_count = round(sequence.shape[0] * 1.1)
    source_positions = np.linspace(0.0, 1.0, sequence.shape[0])
    target_positions = np.linspace(0.0, 1.0, output_count)
    points = sequence.reshape(sequence.shape[0], -1, 4)
    expected = np.empty((output_count, points.shape[1], 4), dtype=np.float32)
    nearest = np.rint(target_positions * (sequence.shape[0] - 1)).astype(int)
    for point_index in range(points.shape[1]):
        for coordinate in range(3):
            expected[:, point_index, coordinate] = np.interp(
                target_positions, source_positions, points[:, point_index, coordinate],
            )
        expected[:, point_index, 3] = points[nearest, point_index, 3]

    np.testing.assert_allclose(actual, expected.reshape(output_count, -1), rtol=1e-6, atol=1e-6)
