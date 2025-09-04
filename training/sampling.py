import torch
import torch.nn.functional as F

__all__ = ["sample_from_logits"]

def sample_from_logits(logits: torch.Tensor, greedy: bool, temperature: float) -> torch.Tensor:
    """Sample indices from logits with optional temperature and greedy mode."""
    if greedy or temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / max(1e-6, temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
