import torch
import torch.nn as nn
from sympy.physics.units import temperature
from torch.nn import functional as F
from .attention import Block
from .utils import precompute_rope_freqs


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, n_head, n_layer, dropout, device, profile) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device

        self.embedding_table = nn.Embedding(vocab_size, n_embed)
        # self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[
            Block(n_embed, n_heads=n_head, block_size=block_size, dropout=dropout, profile=profile)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lmhead = nn.Linear(n_embed, vocab_size)

        # Precompute RoPE freqs for a larger sequence length.
        # Absolute positions are used to ensure cached and non-cached results match.
        self.max_gen_len = 2048 
        self.freqs_complex = precompute_rope_freqs(n_embed // n_head, self.max_gen_len, device=device).to(device)
        self.logs_shown=False

    def forward(self, idx, targets=None, use_cache=False, pos_offset=0, loss_mask=None):
        B, T = idx.shape

        token_emb = self.embedding_table(idx)  # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=self.device) + pos_offset)  # (T, C)
        # x = token_emb + pos_emb  # (B, T, C)
        x = token_emb

        # Get appropriate freqs for the current sequence/cache position

        freqs_complex = self.freqs_complex[pos_offset:pos_offset + T]

        for block in self.blocks:
            x = block(x, use_cache=use_cache, freqs_complex=freqs_complex)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lmhead(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.reshape(B * T)
            if loss_mask is not None:
                loss = F.cross_entropy(logits, targets, reduction='none')
                mask_flat = loss_mask.reshape(B * T)
                loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(logits, targets)

        return logits, loss

    def reset_cache(self):
        for block in self.blocks:
            for head in block.sa.heads:
                head.kv.clear()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, use_cache=False, temperature=0.5, top_k=65, repetition_penalty=1, stop_token_ids=None):
        self.eval()

        def _hit_stop(sequence):
            """Return True if the last tokens of sequence match stop_token_ids."""
            if stop_token_ids is None or len(stop_token_ids) == 0:
                return False
            n = len(stop_token_ids)
            if sequence.shape[1] < n:
                return False
            return sequence[0, -n:].tolist() == stop_token_ids

        if not use_cache:
            pos_offset = 0
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]

                logits, _ = self(idx_cond, use_cache=False, pos_offset=pos_offset)
                pos_offset += 1
                logits_last = logits[:, -1, :]
                logits_last = self.__perform_repetition_penalty(logits_last, idx, repetition_penalty=repetition_penalty)
                logits_last = self.__perform_top_k_filtering(logits_last, top_k=top_k)
                next_idx = self.__perform_temperature_scaling(logits_last, temperature=temperature)

                self.logs_shown = True
                idx = torch.cat((idx, next_idx), dim=1)

                if _hit_stop(idx):
                    break

            return idx

        self.reset_cache()

        # Initial prefill
        idx_cond = idx[:, -self.block_size:]
        current_L = idx.shape[1]
        prefill_pos_offset = max(0, current_L - self.block_size)
        
        logits, _ = self(
            idx_cond,
            use_cache=True,
            pos_offset=prefill_pos_offset
        )

        # cache_pos is the absolute position index of the LAST token we just processed
        cache_pos = current_L - 1

        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :]
            logits_last = self.__perform_repetition_penalty(logits_last, idx, repetition_penalty=repetition_penalty)
            logits_last = self.__perform_top_k_filtering(logits_last, top_k=top_k)
            next_idx = self.__perform_temperature_scaling(logits_last, temperature=temperature)
            self.logs_shown = True
            idx = torch.cat((idx, next_idx), dim=1)

            if _hit_stop(idx):
                break

            cache_pos += 1
            logits, _ = self(
                next_idx,
                use_cache=True,
                pos_offset=cache_pos
            )

        return idx
    def __perform_temperature_scaling(self, logits, temperature):
        if temperature == 0.0:
            next_idx= torch.argmax(logits, dim=-1,keepdim=True)
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            if not self.logs_shown:
                print(f"probs: {probs}")
            next_idx = torch.multinomial(probs, num_samples=1)

        if not self.logs_shown:
            print(f"Temperature: {temperature}")


        return next_idx

    def __perform_top_k_filtering(self,logits: torch.Tensor, top_k):
        max_prediction_token_size = logits.shape[-1]
        if top_k >max_prediction_token_size:
            top_k = max_prediction_token_size

        top_k_logits,_= torch.topk(logits, top_k)

        cutoff = top_k_logits[:,-1].unsqueeze(1)

        logits= logits.masked_fill(logits < cutoff, float("-inf"))

        if not self.logs_shown:
            print(f"Top-k: {top_k}, cutoff: {cutoff.item()}, logits: {logits.shape}")
        return logits

    def __perform_repetition_penalty(self, logits, predicted_tokens,repetition_penalty):
        if repetition_penalty == 0.0:
            return logits
        batch_size, vocab_size = logits.shape

        for batch in range(batch_size):
            for token in predicted_tokens[batch]:
                if logits[batch, token] > 0:
                    logits[batch, token] /=repetition_penalty
                else:
                    logits[batch, token] *=repetition_penalty
        return logits

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
