"""Supervised Fine-Tuning (SFT) training script.

Loads a pre-trained BigramLanguageModel checkpoint and fine-tunes it on
instruction-response pairs derived from the Shakespeare corpus. Loss is
computed only on response tokens via a loss mask.

Usage:
    python sft_train.py
    python sft_train.py --weights path/to/model_weights.pth
"""

import os
import glob
import argparse
from datetime import datetime

import torch

from config import (
    batch_size, block_size, n_embed, n_head, n_layer, dropout, device,
    sft_max_iters, sft_learning_rate, sft_eval_interval, sft_eval_iters,
    sft_source_len, sft_early_stop_patience, sft_overfit_gap,
)
from gpt_modules.model import BigramLanguageModel
from gpt_modules.utils import TimingProfiler
from sft_data_loader import ModernizationSFTDataLoader, SFTDataLoader, CorpusSFTDataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_sft_loss(model: BigramLanguageModel, data_loader: SFTDataLoader):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(sft_eval_iters)
        for k in range(sft_eval_iters):
            x, y, mask = data_loader.get_batch(split)
            # Truncate to block_size if needed
            x, y, mask = x[:, :block_size], y[:, :block_size], mask[:, :block_size]
            _, loss = model(x, y, use_cache=False, loss_mask=mask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def find_latest_weights() -> str | None:
    """Return the most recently modified .pth file in the project directory."""
    candidates = glob.glob(os.path.join(os.path.dirname(__file__), "model_weights_*.pth"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


class EarlyStopping:
    """Stops training when overfitting begins.

    Two triggers:
      1. Val loss has not improved by at least `min_delta` for `patience`
         consecutive eval intervals.
      2. The gap (val_loss - train_loss) exceeds `overfit_gap`, meaning the
         model is fitting the training set much better than the val set.

    The best model state (lowest val loss seen) is saved and can be restored
    after training ends.
    """

    def __init__(self, patience: int, min_delta: float = 1e-3,
                 overfit_gap: float | None = None):
        self.patience = patience
        self.min_delta = min_delta
        self.overfit_gap = overfit_gap

        self.best_val = float("inf")
        self.best_state: dict | None = None
        self.no_improve_count = 0

    def step(self, train_loss: float, val_loss: float, model) -> tuple[bool, str]:
        """Check stopping conditions. Returns (should_stop, reason)."""
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            # Snapshot the best weights (on CPU to save GPU memory)
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            return True, (
                f"val loss has not improved for {self.patience} eval intervals "
                f"(best val: {self.best_val:.4f})"
            )

        if self.overfit_gap is not None and (val_loss - train_loss) > self.overfit_gap:
            return True, (
                f"overfitting gap val-train={val_loss - train_loss:.4f} "
                f"exceeded threshold {self.overfit_gap:.4f}"
            )

        return False, ""

    def restore_best(self, model) -> None:
        """Load the best-seen weights back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT training for GPT-From-Scratch")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to pre-trained weights (.pth). Defaults to the most recent model_weights_*.pth"
    )
    parser.add_argument(
        "--data", type=str, default="./data/input.txt",
        help="Path to the text corpus used for pre-training"
    )
    parser.add_argument(
        "--corpus", type=str, default=None,
        help="Path to corpus.csv (original/translation columns) for real SFT pairs"
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    with open(args.data, "r") as f:
        text = f.read()

    data_loader: ModernizationSFTDataLoader | CorpusSFTDataLoader
    if args.corpus:
        data_loader = CorpusSFTDataLoader(
            csv_path=args.corpus,
            pretrain_text=text,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
        )
    else:
        data_loader = ModernizationSFTDataLoader(
            text=text,
            batch_size=batch_size,
            source_len=sft_source_len,
            device=device,
        )

    print(f"SFT dataset: {len(data_loader.train_inputs)} train / "
          f"{len(data_loader.val_inputs)} val examples")
    print(f"Sequence length: {data_loader.train_inputs.shape[1]} tokens")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    profile = TimingProfiler()
    model = BigramLanguageModel(
        vocab_size=data_loader.vocab_size,
        n_embed=n_embed,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device,
        profile=profile,
    ).to(device)

    weights_path = args.weights or find_latest_weights()
    if weights_path and os.path.exists(weights_path):
        print(f"Loading pre-trained weights: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Warning: no pre-trained weights found — fine-tuning from random init.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_learning_rate)

    print(f"\nStarting SFT on device: {device}")
    print(f"  max_iters={sft_max_iters}  lr={sft_learning_rate}  "
          f"source_len={sft_source_len}")
    print(f"  early_stop_patience={sft_early_stop_patience}  "
          f"overfit_gap={sft_overfit_gap}\n")

    early_stop = EarlyStopping(
        patience=sft_early_stop_patience,
        overfit_gap=sft_overfit_gap,
    )

    model.train()
    stopped_early = False
    for step in range(sft_max_iters):
        if step % sft_eval_interval == 0:
            losses = estimate_sft_loss(model, data_loader)
            gap = losses['val'] - losses['train']
            print(f"step {step:4d}: train loss {losses['train']:.4f}  "
                  f"val loss {losses['val']:.4f}  gap {gap:+.4f}")

            should_stop, reason = early_stop.step(
                losses['train'].item(), losses['val'].item(), model
            )
            if should_stop:
                print(f"\nEarly stopping at step {step}: {reason}")
                stopped_early = True
                break

        x, y, mask = data_loader.get_batch("train")
        x    = x[:, :block_size]
        y    = y[:, :block_size]
        mask = mask[:, :block_size]

        _, loss = model(x, y, use_cache=False, loss_mask=mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if not stopped_early:
        losses = estimate_sft_loss(model, data_loader)
        print(f"\nFinal — train loss {losses['train']:.4f}  val loss {losses['val']:.4f}")

    # Restore the best checkpoint before saving
    early_stop.restore_best(model)
    print(f"Restoring best weights (val loss: {early_stop.best_val:.4f})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"sft_weights_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nFine-tuned weights saved to: {save_path}")


if __name__ == "__main__":
    main()
