import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
from .utils import KVCache, TimingProfiler, apply_rope


# We'll pass hyperparameters as arguments to the classes to avoid circular imports or dependency on a global config
# but for now, let's assume they are passed during initialization.

class Head(nn.Module):
    """single headed attention"""

    def __init__(self, n_embed, head_size, block_size, dropout, profile):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.kv = KVCache(max_seq_len=block_size)
        self.profile = profile

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache=False, freqs_complex=None):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if freqs_complex is not None:
            q = apply_rope(q, freqs_complex, device=x.device)
            k = apply_rope(k, freqs_complex, device=x.device)

        if use_cache:
            start = datetime.now()
            k, v = self.kv.update(k, v)
            end = datetime.now()
            self.profile.record("KVCache", (end - start).total_seconds())

            start = datetime.now()
            wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

            if T > 1:
                total_T = k.shape[1]
                # During prefill, the current tokens T can see all previous history in the cache
                # and follow causal masking within the chunk T.
                mask = torch.tril(torch.ones(T, T, device=x.device))
                # Pad mask to match k sequence length (total_T)
                full_mask = torch.cat([
                    torch.ones(T, total_T - T, device=x.device),
                    mask
                ], dim=1)
                wei = wei.masked_fill(full_mask == 0, float("-inf"))

            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)

            out = wei @ v
            end = datetime.now()
            self.profile.record("Attention", (end - start).total_seconds())
            return out

        else:
            start = datetime.now()
            wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)

            out = wei @ v
            end = datetime.now()
            self.profile.record("Attention", (end - start).total_seconds())
            return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, n_embed, num_heads, head_size, block_size, dropout, profile) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout, profile) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache, freqs_complex=None):
        out = torch.cat([h(x, use_cache=use_cache, freqs_complex=freqs_complex) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """simple linear layer followed by a non-linear activation"""

    def __init__(self, n_embed, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_heads, block_size, dropout, profile) -> None:
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_embed, n_heads, head_size, block_size, dropout, profile)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, use_cache=False, freqs_complex=None):
        x = x + self.sa(self.ln1(x), use_cache, freqs_complex=freqs_complex)
        x = x + self.ffwd(self.ln2(x))
        return x
