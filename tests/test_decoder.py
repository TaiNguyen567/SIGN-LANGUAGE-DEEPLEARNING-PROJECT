import pytest
import torch

from src.models.ctc_decoder import CTCGreedyDecoder


def test_greedy_ctc_collapses_repeats_but_keeps_blank_separated_repeat():
    token_path = torch.tensor([[0], [2], [2], [0], [2], [3], [3], [0]])
    log_probs = torch.full((8, 1, 4), -8.0)
    log_probs.scatter_(2, token_path.unsqueeze(-1), 0.0)

    decoded = CTCGreedyDecoder(blank_id=0).decode(log_probs)

    assert decoded[0].token_ids == (2, 2, 3)
    assert decoded[0].confidence > 0.9


def test_decoder_respects_input_lengths_and_empty_path():
    log_probs = torch.log_softmax(torch.zeros(5, 2, 3), dim=-1)
    decoded = CTCGreedyDecoder().decode(log_probs, torch.tensor([0, 3]))

    assert decoded[0].token_ids == ()
    assert decoded[0].confidence == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_greedy_decoder_handles_cuda_without_changing_ctc_path():
    token_path = torch.tensor([[0], [2], [2], [0], [2], [3], [3], [0]])
    log_probs = torch.full((8, 1, 4), -8.0)
    log_probs.scatter_(2, token_path.unsqueeze(-1), 0.0)

    decoded = CTCGreedyDecoder(blank_id=0).decode(log_probs.cuda())

    assert decoded[0].token_ids == (2, 2, 3)
    assert decoded[0].confidence > 0.9


def test_decoder_fallback_recovers_empty_prediction():
    # log_probs with blank dominating slightly (e.g. blank=0.55, class 2=0.45)
    log_probs = torch.tensor([
        [[-0.6, -10.0, -0.8]],  # blank=0.55, class 2=0.45
        [[-0.6, -10.0, -0.8]],
    ])  # [T=2, B=1, C=3]

    # Without fallback -> empty
    decoded_no_fb = CTCGreedyDecoder(blank_id=0, fallback_non_blank=False).decode(log_probs)
    assert decoded_no_fb[0].token_ids == ()

    # With fallback -> recovers class 2
    decoded_fb = CTCGreedyDecoder(blank_id=0, fallback_non_blank=True).decode(log_probs)
    assert decoded_fb[0].token_ids == (2,)
    assert decoded_fb[0].confidence > 0.4

