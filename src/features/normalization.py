"""Normalize landmark groups and attach per-landmark visibility masks."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class LandmarkNormalizer:
    """Center and scale landmark coordinates using stable anatomical anchors."""

    def normalize(
        self,
        landmarks: np.ndarray | None,
        *,
        kind: str,
        expected_count: int,
        valid_mask: np.ndarray | None = None,
        reference_indices: Sequence[int] = (),
        scale_indices: Sequence[int] = (),
    ) -> np.ndarray:
        """Return rows of ``x, y, z, valid`` with missing points set to zero."""
        if expected_count < 0:
            raise ValueError("expected_count must be non-negative")

        coords = np.zeros((expected_count, 3), dtype=np.float32)
        present = np.zeros(expected_count, dtype=bool)
        if landmarks is not None:
            source = np.asarray(landmarks, dtype=np.float32)
            if source.ndim != 2 or source.shape[1] < 3:
                raise ValueError("landmarks must have shape [N, 3] or [N, >=3]")
            count = min(expected_count, source.shape[0])
            coords[:count] = source[:count, :3]
            present[:count] = np.isfinite(coords[:count]).all(axis=1)

        if valid_mask is not None:
            supplied_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
            count = min(expected_count, supplied_mask.size)
            present[count:] = False
            present[:count] &= supplied_mask[:count]
        coords[~present] = 0.0

        if not present.any():
            return np.zeros((expected_count, 4), dtype=np.float32)

        reference_points = self._valid_indices(reference_indices, present)
        if reference_points:
            center = coords[reference_points].mean(axis=0)
        else:
            center = coords[present].mean(axis=0)

        scale_points = self._valid_indices(scale_indices, present)
        if len(scale_points) >= 2:
            scale = float(np.linalg.norm(coords[scale_points[0], :2] - coords[scale_points[1], :2]))
        else:
            visible_xy = coords[present, :2]
            scale = float(np.linalg.norm(visible_xy.max(axis=0) - visible_xy.min(axis=0)))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        normalized = np.zeros_like(coords)
        normalized[present] = (coords[present] - center) / scale
        return np.concatenate((normalized, present[:, None].astype(np.float32)), axis=1)

    @staticmethod
    def _valid_indices(indices: Sequence[int], present: np.ndarray) -> list[int]:
        return [index for index in indices if 0 <= index < present.size and present[index]]
