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
            token_counts: list[int] = []
            previous = self.blank_id
            for frame_id, frame_confidence in packed_path[batch_index][:limit]:
                current = int(frame_id)
                conf = float(frame_confidence)
                if current != self.blank_id:
                    if current != previous:
                        token_ids.append(current)
                        token_confidences.append(conf)
                        token_counts.append(1)
                    elif token_confidences:
                        token_confidences[-1] = max(token_confidences[-1], conf)
                        token_counts[-1] += 1
                previous = current

            if len(token_ids) > 1:
                max_conf = max(token_confidences)
                max_count = max(token_counts)
                filtered_pairs = [
                    (tid, tconf)
                    for tid, tconf, tcnt in zip(token_ids, token_confidences, token_counts)
                    if not (tcnt == 1 and max_count >= 2 and tconf < 0.75 and tconf < max_conf * 0.8)
                ]
                if filtered_pairs:
                    token_ids = [p[0] for p in filtered_pairs]
                    token_confidences = [p[1] for p in filtered_pairs]

            if not token_ids and self.fallback_non_blank and limit > 0 and classes > 1:
                seq_probs = probabilities[:limit, batch_index].detach().clone()
                seq_probs[:, self.blank_id] = 0.0
                if classes > 2:
                    seq_probs[:, 1] = 0.0
                top_k = min(5, limit)
                topk_vals, _ = torch.topk(seq_probs, k=top_k, dim=0)
                mean_topk = topk_vals.mean(dim=0)
                best_token = int(torch.argmax(mean_topk).item())
                best_conf = float(seq_probs[:, best_token].max().item())
                if best_conf >= self.min_fallback_confidence:
                    token_ids.append(best_token)
                    token_confidences.append(best_conf)

            confidence = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
            outputs.append(DecodedOutput(tuple(token_ids), confidence))
        return outputs
