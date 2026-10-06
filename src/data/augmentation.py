"""Conservative spatial and temporal augmentation for cached landmarks."""

from __future__ import annotations

from typing import Any

import numpy as np


class LandmarkAugmenter:
    """Augment ``[T, F]`` arrays whose point layout is ``[x, y, z, valid]``."""

    def __init__(self, config: dict[str, Any], *, seed: int | None = None) -> None:
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.rng = np.random.default_rng(seed)

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        values = np.asarray(sequence, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] % 4 != 0:
            raise ValueError("Landmark augmentation expects shape [T, 4 * number_of_points]")
        if not self.enabled:
            return values.copy()

        values = self._temporal_augment(values)
        points = values.reshape(values.shape[0], -1, 4).copy()
        coords = points[:, :, :3]
        visible = points[:, :, 3] > 0.5

        noise_std = float(self.config.get("gaussian_noise_std", 0.0))
        if noise_std > 0:
            coords += self.rng.normal(0.0, noise_std, size=coords.shape).astype(np.float32) * visible[:, :, None]

        scale_range = self.config.get("scale_range", (1.0, 1.0))
        low, high = float(scale_range[0]), float(scale_range[1])
        if low <= 0 or high < low:
            raise ValueError("scale_range must contain positive ascending bounds")
        coords *= float(self.rng.uniform(low, high))

        translation_std = float(self.config.get("translation_std", 0.0))
        if translation_std > 0:
            shift = self.rng.normal(0.0, translation_std, size=(1, 1, 3)).astype(np.float32)
            coords += shift * visible[:, :, None]

        max_rotation = float(self.config.get("rotation_degrees", 0.0))
        if max_rotation > 0:
            angle = np.deg2rad(self.rng.uniform(-max_rotation, max_rotation))
            cosine, sine = np.cos(angle), np.sin(angle)
            x_values, y_values = coords[:, :, 0].copy(), coords[:, :, 1].copy()
            coords[:, :, 0] = cosine * x_values - sine * y_values
            coords[:, :, 1] = sine * x_values + cosine * y_values

        mask_probability = float(self.config.get("landmark_mask_probability", 0.0))
        if mask_probability > 0:
            hidden = self.rng.random(visible.shape) < mask_probability
            visible &= ~hidden
            coords[~visible] = 0.0
            points[:, :, 3] = visible.astype(np.float32)

        coords[~visible] = 0.0
        return points.reshape(values.shape[0], -1).astype(np.float32, copy=False)

    def _temporal_augment(self, sequence: np.ndarray) -> np.ndarray:
        values = sequence
        if values.shape[0] > 4 and self.rng.random() < float(self.config.get("temporal_crop_probability", 0.0)):
            keep_ratio = self.rng.uniform(0.85, 1.0)
            keep_count = max(2, int(round(values.shape[0] * keep_ratio)))
            start = int(self.rng.integers(0, values.shape[0] - keep_count + 1))
            values = values[start:start + keep_count]

        if values.shape[0] > 2 and self.rng.random() < float(self.config.get("temporal_stretch_probability", 0.0)):
            factor = float(self.rng.uniform(0.9, 1.1))
            output_count = max(2, int(round(values.shape[0] * factor)))
            source_positions = np.linspace(0.0, 1.0, values.shape[0])
            target_positions = np.linspace(0.0, 1.0, output_count)
            points = values.reshape(values.shape[0], -1, 4)
            stretched = np.empty((output_count, points.shape[1], 4), dtype=np.float32)
            upper = np.searchsorted(source_positions, target_positions, side="right")
            upper = np.clip(upper, 1, values.shape[0] - 1)
            lower = upper - 1
            weight = (
                (target_positions - source_positions[lower])
                / (source_positions[upper] - source_positions[lower])
            )
            coordinates = points[:, :, :3].reshape(values.shape[0], -1)
            stretched[:, :, :3] = (
                coordinates[lower] * (1.0 - weight[:, None])
                + coordinates[upper] * weight[:, None]
            ).reshape(output_count, points.shape[1], 3)
            nearest = np.rint(target_positions * (values.shape[0] - 1)).astype(int)
            stretched[:, :, 3] = points[nearest, :, 3]
            values = stretched.reshape(output_count, -1)

        drop_probability = float(self.config.get("frame_drop_probability", 0.0))
        if values.shape[0] > 2 and drop_probability > 0:
            keep = self.rng.random(values.shape[0]) >= drop_probability
            if keep.sum() < 2:
                keep[self.rng.choice(values.shape[0], size=2, replace=False)] = True
            values = values[keep]
        return values
