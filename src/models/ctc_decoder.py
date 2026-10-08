"""Greedy CTC decoding with token confidence estimates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecodedOutput:
    token_ids: tuple[int, ...]
    confidence: float


class CTCGreedyDecoder:
    """Collapse repeated labels and blanks in the most likely CTC path."""

    def __init__(
        self,
        blank_id: int = 0,
        *,
        fallback_non_blank: bool = True,
        min_fallback_confidence: float = 0.001,
    ) -> None:
        if blank_id < 0:
            raise ValueError("blank_id must be non-negative")
        self.blank_id = blank_id
        self.fallback_non_blank = fallback_non_blank
        self.min_fallback_confidence = min_fallback_confidence

    def decode(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> list[DecodedOutput]:
        """Decode model output with shape ``[T, B, C]``."""
        if log_probs.ndim != 3:
            raise ValueError("log_probs must have shape [time, batch, classes]")
        time_steps, batch_size, classes = log_probs.shape
        if self.blank_id >= classes:
            raise ValueError("blank_id is outside the model class dimension")
        if lengths is None:
            limits = [time_steps] * batch_size
        else:
            limits = [int(value) for value in torch.as_tensor(lengths).reshape(-1).tolist()]
            if len(limits) != batch_size or any(limit < 0 or limit > time_steps for limit in limits):
                raise ValueError("lengths must contain valid frame counts for every batch item")

        probabilities = log_probs.detach().float().exp()
        best_ids = log_probs.detach().argmax(dim=-1)
        best_probabilities = probabilities.gather(-1, best_ids.unsqueeze(-1)).squeeze(-1)
        packed_path = torch.stack(
            (best_ids.to(dtype=best_probabilities.dtype), best_probabilities), dim=-1,
        ).transpose(0, 1).cpu().tolist()
        outputs: list[DecodedOutput] = []
        for batch_index, limit in enumerate(limits):
            token_ids: list[int] = []
            token_confidences: list[float] = []
            previous = self.blank_id
            for frame_id, frame_confidence in packed_path[batch_index][:limit]:
                current = int(frame_id)
                if current != self.blank_id and current != previous:
                    token_ids.append(current)
                    token_confidences.append(float(frame_confidence))
                previous = current

            if not token_ids and self.fallback_non_blank and limit > 0 and classes > 1:
                seq_lp = log_probs[:limit, batch_index].detach().clone()
                seq_lp[:, self.blank_id] = -float("inf")
                max_val, max_idx = torch.max(seq_lp.reshape(-1), dim=0)
                best_token = (max_idx % classes).item()
                best_conf = float(torch.exp(max_val).item())
                if best_conf >= self.min_fallback_confidence:
                    token_ids.append(best_token)
                    token_confidences.append(best_conf)

            confidence = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
            outputs.append(DecodedOutput(tuple(token_ids), confidence))
        return outputs
