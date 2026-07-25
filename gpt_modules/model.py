import torch
import torch.nn as nn
from torch.nn import functional as F
from .attention import Block

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, n_head, n_layer, dropout, device, profile) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device
        
        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[
            Block(n_embed, n_heads=n_head, block_size=block_size, dropout=dropout, profile=profile) 
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lmhead = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None, use_cache=False, pos_offset=0):
        B, T = idx.shape

        token_emb = self.embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device) + pos_offset)  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lmhead(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def reset_cache(self):
        for block in self.blocks:
            for head in block.sa.heads:
                head.kv.clear()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=False):
        self.eval()

        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits, _ = self(idx_cond, use_cache=False, pos_offset=0)

                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)

                idx = torch.cat((idx, next_idx), dim=1)

            return idx

        self.reset_cache()

        # Initial prefill
        idx_cond = idx[:, -self.block_size:]
        logits, _ = self(
            idx_cond,
            use_cache=True,
            pos_offset=0
        )

        cache_pos = idx_cond.shape[1] - 1

        for _ in range(max_new_tokens):

            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)

            prev_idx = idx[:, -1:]
            idx = torch.cat((idx, next_idx), dim=1)

            if cache_pos < self.block_size - 1:
                cache_pos += 1
                logits, _ = self(
                    next_idx,
                    use_cache=True,
                    pos_offset=cache_pos
                )
            else:
                self.reset_cache()
                logits, _ = self(
                    prev_idx,
                    use_cache=True,
                    pos_offset=0
                )
                logits, _ = self(
                    next_idx,
                    use_cache=True,
                    pos_offset=1
                )
                cache_pos = 1

        return idx

    @torch.no_grad()
    def compare_cache(self, idx):
        self.eval()

        # -------------------------
        # Non-cached forward
        # -------------------------
        self.reset_cache()

        logits_no_cache, _ = self(
            idx,
            use_cache=False,
            pos_offset=0
        )

        # -------------------------
        # Cached forward
        # -------------------------
        self.reset_cache()

        logits_cache, _ = self(
            idx,
            use_cache=True,
            pos_offset=0
        )

        difference = torch.max(
            torch.abs(
                logits_no_cache - logits_cache
            )
        )

        print(
            "No-cache shape:",
            logits_no_cache.shape
        )

        print(
            "Cache shape:",
            logits_cache.shape
        )

        print(
            "Maximum difference:",
            difference.item()
        )

        return logits_no_cache, logits_cache
