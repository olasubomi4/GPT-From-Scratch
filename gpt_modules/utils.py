import torch
from collections import defaultdict

class TimingProfiler:
    def __init__(self):
        self.timings = defaultdict(list)

    def record(self, name: str, elapsed: float):
        self.timings[name].append(elapsed)

    def summary(self):
        print("\n===== PROFILING SUMMARY =====")

        total_runtime = 0

        for name, values in self.timings.items():
            total = sum(values)
            avg = total / len(values)
            maximum = max(values)

            total_runtime += total

            print(
                f"{name:<20} "
                f"calls={len(values):<8} "
                f"total={total:.6f}s "
                f"avg={avg:.6f}s "
                f"max={maximum:.6f}s"
            )

        print(f"\nTotal measured time: {total_runtime:.6f}s")


class KVCache:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

        self.cache_k = None
        self.cache_v = None

        self.cur_len = 0

    def update(self, new_k, new_v):
        B, T, C = new_k.shape

        # First allocation
        if self.cache_k is None:
            self.cache_k = torch.zeros(
                B,
                256,
                C,
                device=new_k.device,
                dtype=new_k.dtype
            )

            self.cache_v = torch.zeros(
                B,
                256,
                C,
                device=new_v.device,
                dtype=new_v.dtype
            )

            self.cur_len = 0

        for t in range(T):

            # Cache not full
            if self.cur_len < 256:
                self.cache_k[:, self.cur_len:self.cur_len + 1] = new_k[:, t:t + 1]
                self.cache_v[:, self.cur_len:self.cur_len + 1] = new_v[:, t:t + 1]
                self.cur_len += 1

            # Sliding window
            else:
                self.cache_k[:, :-1] = self.cache_k[:, 1:].clone()
                self.cache_v[:, :-1] = self.cache_v[:, 1:].clone()

                self.cache_k[:, -1:] = new_k[:, t:t + 1]
                self.cache_v[:, -1:] = new_v[:, t:t + 1]

        return (
            self.cache_k[:, :self.cur_len],
            self.cache_v[:, :self.cur_len]
        )

    def clear(self):
        self.cache_k = None
        self.cache_v = None
        self.cur_len = 0


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0, device: str = 'cpu'):
    """
    Precompute the frequencies for rotary positional embeddings.
    """
    # Half of the dimension is used for cosine and sine
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs).to(device)  # (max_seq_len, dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)  # (max_seq_len, dim/2)
    return freqs_complex


def apply_rope(x: torch.Tensor, freqs_complex: torch.Tensor, device: str = 'cpu'):
    """
    Apply rotary positional embeddings to a tensor.
    """
    # x: (B, T, head_size)
    # freqs_complex: (T, head_size/2)

    # Reshape x to complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (B, T, head_size/2)

    # Broadcast freqs_complex to match B
    freqs_complex = freqs_complex.unsqueeze(0)  # (1, T, head_size/2)

    # Multiply
    x_rotated = x_complex * freqs_complex  # (B, T, head_size/2)

    out = torch.view_as_real(x_rotated).reshape(*x.shape)  # (B, T, head_size)
    return out.type_as(x)
