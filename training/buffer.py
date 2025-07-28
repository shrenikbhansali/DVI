import logging
from typing import Dict, Optional

import torch

__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    """Fixed-capacity token replay buffer stored on a single device."""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device

        # Storage buffers
        self._hidden_buf: Optional[torch.Tensor] = None
        self._token_buf = torch.empty(self.capacity, dtype=torch.long, device=device)
        self._reward_buf = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self._conf_buf = torch.empty(self.capacity, dtype=torch.float32, device=device)

        # Pointers
        self._write_idx = 0
        self._size = 0

    @torch.no_grad()
    def append(
        self,
        hidden: torch.Tensor,
        token: int,
        reward: float,
        conf: float,
    ) -> bool:
        """Append a transition and return ``True`` if an old slot was overwritten."""
        if self._hidden_buf is None:
            shape = (self.capacity,) + tuple(hidden.shape)
            self._hidden_buf = torch.empty(shape, dtype=hidden.dtype, device=self.device)
            logging.debug("ReplayBuffer allocated with shape %s", str(shape))

        idx = self._write_idx
        drop = self._size == self.capacity
        if drop:
            logging.debug("Overwriting index %d with token %d", idx, int(token))

        self._hidden_buf[idx].copy_(hidden.detach().to(self.device))
        self._token_buf[idx] = int(token)
        self._reward_buf[idx] = float(reward)
        self._conf_buf[idx] = float(conf)

        self._write_idx = (idx + 1) % self.capacity
        if not drop:
            self._size += 1
        return drop

    @torch.no_grad()
    def sample(self, batch_size: int, accepted_only: bool = True) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions uniformly."""
        if self._size == 0 or self._hidden_buf is None:
            raise ValueError("Buffer is empty")

        mask = self._reward_buf[: self._size] == 1.0
        if not accepted_only:
            mask = torch.ones_like(mask, dtype=torch.bool)

        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        count = int(indices.numel())
        if batch_size > count:
            raise ValueError("Not enough samples of requested type")

        perm = torch.randperm(count, device=self.device)[:batch_size]
        choice = indices[perm]
        logging.debug("Sampling %d entries (accepted_only=%s)", batch_size, accepted_only)

        return {
            "hidden": self._hidden_buf[choice].clone(),
            "token": self._token_buf[choice].clone(),
            "reward": self._reward_buf[choice].clone(),
            "conf": self._conf_buf[choice].clone(),
        }

    @torch.no_grad()
    def accepted_count(self) -> int:
        if self._size == 0:
            return 0
        return int((self._reward_buf[: self._size] == 1.0).sum().item())

    def __len__(self) -> int:
        return self._size

    @torch.no_grad()
    def clear(self, accepted_only: bool = False) -> None:
        if not accepted_only:
            self._size = 0
            self._write_idx = 0
            return

        if self._size == 0:
            return

        keep_mask = self._reward_buf[: self._size] != 1.0
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        keep_count = int(keep_indices.numel())

        if keep_count:
            self._hidden_buf[:keep_count].copy_(self._hidden_buf[keep_indices])
            self._token_buf[:keep_count].copy_(self._token_buf[keep_indices])
            self._reward_buf[:keep_count].copy_(self._reward_buf[keep_indices])
            self._conf_buf[:keep_count].copy_(self._conf_buf[keep_indices])

        self._size = keep_count
        self._write_idx = keep_count % self.capacity
