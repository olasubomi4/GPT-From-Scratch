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
                self.max_seq_len,
                C,
                device=new_k.device,
                dtype=new_k.dtype
            )

            self.cache_v = torch.zeros(
                B,
                self.max_seq_len,
                C,
                device=new_v.device,
                dtype=new_v.dtype
            )

            self.cur_len = 0

        for t in range(T):

            # Cache not full
            if self.cur_len < self.max_seq_len:
                self.cache_k[:, self.cur_len:self.cur_len+1] = new_k[:, t:t+1]
                self.cache_v[:, self.cur_len:self.cur_len+1] = new_v[:, t:t+1]
                self.cur_len += 1

            # Sliding window
            else:
                self.cache_k[:, :-1] = self.cache_k[:, 1:].clone()
                self.cache_v[:, :-1] = self.cache_v[:, 1:].clone()

                self.cache_k[:, -1:] = new_k[:, t:t+1]
                self.cache_v[:, -1:] = new_v[:, t:t+1]

        return (
            self.cache_k[:, :self.cur_len],
            self.cache_v[:, :self.cur_len]
        )

    def clear(self):
        self.cache_k = None
        self.cache_v = None
        self.cur_len = 0
