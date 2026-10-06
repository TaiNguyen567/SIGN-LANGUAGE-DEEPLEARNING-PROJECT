import torch

from src.models.transformer import SignLanguageTransformer
from src.training.losses import CTCLoss


def test_transformer_forward_and_ctc_backward():
    model = SignLanguageTransformer(
        input_dim=12, num_classes=5, d_model=32, nhead=4,
        num_layers=1, dim_feedforward=64, dropout=0.0, max_len=16,
    )
    features = torch.randn(2, 8, 12)
    lengths = torch.tensor([8, 6])

    log_probs = model(features, lengths)

    assert log_probs.shape == (8, 2, 5)
    torch.testing.assert_close(log_probs.exp().sum(dim=-1), torch.ones(8, 2), atol=1e-5, rtol=1e-5)
    loss = CTCLoss()(log_probs, torch.tensor([1, 2, 3]), lengths, torch.tensor([2, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.classifier.weight.grad is not None
