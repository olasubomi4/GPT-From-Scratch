import torch

from sft_modernizer import modernize, has_archaic


INSTRUCTION_TEMPLATE = (
    "Instruction:\n"
    "Continue this passage:\n"
    "'{instruction}'\n\n"
    "Response:\n"
    "{response}"
)


class SFTDataLoader:
    """Builds instruction-response pairs from raw text for Supervised Fine-Tuning.

    Each example is formatted with a prompt template. The loss mask is 0 for
    all instruction tokens and 1 for response tokens, so loss is only computed
    on the part the model should learn to generate.
    """

    def __init__(self, text: str, batch_size: int, instruction_len: int,
                 response_len: int, device: str):
        self.batch_size = batch_size
        self.instruction_len = instruction_len
        self.response_len = response_len
        self.device = device

        # Build char-level vocab from the full text (same as pre-training)
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.ctoi = {c: i for i, c in enumerate(chars)}
        self.itoc = {i: c for i, c in enumerate(chars)}

        # Build all (instruction_snippet, response_snippet) pairs
        examples = self._build_examples(text)

        # Encode and create tensors with loss masks
        self.input_ids, self.target_ids, self.loss_masks = self._encode_examples(examples)

        n = int(0.9 * len(self.input_ids))
        self.train_inputs,  self.val_inputs  = self.input_ids[:n],  self.input_ids[n:]
        self.train_targets, self.val_targets = self.target_ids[:n], self.target_ids[n:]
        self.train_masks,   self.val_masks   = self.loss_masks[:n], self.loss_masks[n:]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_batch(self, split: str):
        inputs  = self.train_inputs  if split == "train" else self.val_inputs
        targets = self.train_targets if split == "train" else self.val_targets
        masks   = self.train_masks   if split == "train" else self.val_masks

        ix = torch.randint(len(inputs), (self.batch_size,))
        x    = inputs[ix].to(self.device)
        y    = targets[ix].to(self.device)
        mask = masks[ix].to(self.device)
        return x, y, mask

    def encode(self, s: str):
        return [self.ctoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itoc[i] for i in ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_examples(self, text: str):
        """Slide over the corpus to collect (instruction_snippet, response_snippet) pairs."""
        examples = []
        step = self.response_len  # non-overlapping windows
        i = 0
        while i + self.instruction_len + self.response_len <= len(text):
            inst = text[i: i + self.instruction_len]
            resp = text[i + self.instruction_len: i + self.instruction_len + self.response_len]
            examples.append((inst, resp))
            i += step
        return examples

    def _encode_examples(self, examples):
        """Encode all examples; return stacked input_ids, target_ids, loss_masks."""
        all_inputs, all_targets, all_masks = [], [], []

        for inst_text, resp_text in examples:
            full_text = INSTRUCTION_TEMPLATE.format(
                instruction=inst_text, response=resp_text
            )

            # Determine where the response starts inside the formatted string
            prefix = INSTRUCTION_TEMPLATE.format(
                instruction=inst_text, response=""
            )
            resp_start_char = len(prefix)

            # Skip examples with characters outside our vocabulary
            if not all(c in self.ctoi for c in full_text):
                continue

            ids = self.encode(full_text)
            seq_len = len(ids)

            # Shift by 1: input is ids[:-1], target is ids[1:]
            input_ids  = torch.tensor(ids[:-1], dtype=torch.long)
            target_ids = torch.tensor(ids[1:],  dtype=torch.long)

            # loss_mask: 1 only for positions where the target is a response token
            # resp_start_char is the char index where the response begins in `full_text`.
            # The target at position t corresponds to full_text[t+1], so we mask
            # positions t where t+1 >= resp_start_char.
            loss_mask = torch.zeros(seq_len - 1, dtype=torch.float)
            resp_token_start = resp_start_char - 1  # token index in input_ids
            if resp_token_start < seq_len - 1:
                loss_mask[resp_token_start:] = 1.0

            all_inputs.append(input_ids)
            all_targets.append(target_ids)
            all_masks.append(loss_mask)

        # Pad/truncate all sequences to the same length (max in the batch)
        max_len = max(t.shape[0] for t in all_inputs)
        padded_inputs, padded_targets, padded_masks = [], [], []
        for inp, tgt, msk in zip(all_inputs, all_targets, all_masks):
            pad = max_len - inp.shape[0]
            padded_inputs.append(torch.nn.functional.pad(inp, (0, pad)))
            padded_targets.append(torch.nn.functional.pad(tgt, (0, pad)))
            padded_masks.append(torch.nn.functional.pad(msk, (0, pad)))

        return (
            torch.stack(padded_inputs),
            torch.stack(padded_targets),
            torch.stack(padded_masks),
        )


# ---------------------------------------------------------------------------
# Modernization SFT data loader
# ---------------------------------------------------------------------------

MODERNIZE_TEMPLATE = (
    "Instruction:\n"
    "Rewrite this in modern English:\n"
    "'{source}'\n\n"
    "Response:\n"
    "{response}"
)


class ModernizationSFTDataLoader:
    """SFT data loader for the 'Rewrite in modern English' instruction task.

    Each training example is:
      Instruction: Rewrite this in modern English: '<archaic chunk>'
      Response:    <modernized chunk>

    Loss is computed only on Response tokens via a loss mask.
    """

    def __init__(self, text: str, batch_size: int, source_len: int, device: str):
        self.batch_size = batch_size
        self.source_len = source_len
        self.device = device

        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.ctoi = {c: i for i, c in enumerate(chars)}
        self.itoc = {i: c for i, c in enumerate(chars)}

        pairs = self._build_pairs(text)
        print(f"ModernizationSFTDataLoader: {len(pairs)} archaic→modern pairs found")

        self.input_ids, self.target_ids, self.loss_masks = self._encode_pairs(pairs)

        n = int(0.9 * len(self.input_ids))
        self.train_inputs,  self.val_inputs  = self.input_ids[:n],  self.input_ids[n:]
        self.train_targets, self.val_targets = self.target_ids[:n], self.target_ids[n:]
        self.train_masks,   self.val_masks   = self.loss_masks[:n], self.loss_masks[n:]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_batch(self, split: str):
        inputs  = self.train_inputs  if split == "train" else self.val_inputs
        targets = self.train_targets if split == "train" else self.val_targets
        masks   = self.train_masks   if split == "train" else self.val_masks

        ix = torch.randint(len(inputs), (self.batch_size,))
        return (
            inputs[ix].to(self.device),
            targets[ix].to(self.device),
            masks[ix].to(self.device),
        )

    def encode(self, s: str):
        return [self.ctoi[c] for c in s if c in self.ctoi]

    def decode(self, ids):
        return "".join(self.itoc[i] for i in ids)

    def make_prompt(self, source: str) -> str:
        """Return the instruction prefix (no response) for generation-time use."""
        return (
            "Instruction:\n"
            "Rewrite this in modern English:\n"
            f"'{source}'\n\n"
            "Response:\n"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pairs(self, text: str):
        """Slide over the corpus in steps of source_len, keep chunks with archaic words."""
        pairs = []
        i = 0
        while i + self.source_len <= len(text):
            chunk = text[i: i + self.source_len]
            if has_archaic(chunk):
                pairs.append((chunk, modernize(chunk)))
            i += self.source_len
        return pairs

    def _encode_pairs(self, pairs):
        all_inputs, all_targets, all_masks = [], [], []

        for source, modern in pairs:
            full_text = MODERNIZE_TEMPLATE.format(source=source, response=modern + "\n\n")
            prefix    = MODERNIZE_TEMPLATE.format(source=source, response="")
            resp_start_char = len(prefix)

            if not all(c in self.ctoi for c in full_text):
                continue

            ids = [self.ctoi[c] for c in full_text]
            seq_len = len(ids)

            input_ids  = torch.tensor(ids[:-1], dtype=torch.long)
            target_ids = torch.tensor(ids[1:],  dtype=torch.long)

            loss_mask = torch.zeros(seq_len - 1, dtype=torch.float)
            resp_token_start = resp_start_char - 1
            if resp_token_start < seq_len - 1:
                loss_mask[resp_token_start:] = 1.0

            all_inputs.append(input_ids)
            all_targets.append(target_ids)
            all_masks.append(loss_mask)

        if not all_inputs:
            raise ValueError("No valid modernization pairs found. Check that input.txt contains archaic Shakespeare text.")

        max_len = max(t.shape[0] for t in all_inputs)
        pad = torch.nn.functional.pad
        return (
            torch.stack([pad(x, (0, max_len - x.shape[0])) for x in all_inputs]),
            torch.stack([pad(x, (0, max_len - x.shape[0])) for x in all_targets]),
            torch.stack([pad(x, (0, max_len - x.shape[0])) for x in all_masks]),
        )


# ---------------------------------------------------------------------------
# Corpus-based SFT data loader (reads corpus.csv with original/translation cols)
# ---------------------------------------------------------------------------

class CorpusSFTDataLoader:
    """SFT data loader backed by a real Shakespeare→Modern-English translation corpus.

    Expects a CSV with at least two columns: 'original' and 'translation'.
    Characters outside the pre-training vocab are normalised (smart quotes,
    em-dashes, accents, annotations) and rows that still contain unknown chars
    are dropped.

    Each example is formatted as:
        Instruction:
        Rewrite this in modern English:
        '<original>'

        Response:
        <translation>

    Loss mask is 0 for the instruction/original prefix and 1 for the
    translation tokens only.
    """

    import re as _re

    def __init__(self, csv_path: str, pretrain_text: str,
                 batch_size: int, block_size: int, device: str):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        chars = sorted(set(pretrain_text))
        self.vocab_size = len(chars)
        self.ctoi = {c: i for i, c in enumerate(chars)}
        self.itoc = {i: c for i, c in enumerate(chars)}

        pairs = self._load_pairs(csv_path)
        print(f"CorpusSFTDataLoader: {len(pairs)} usable pairs from {csv_path}")

        self.input_ids, self.target_ids, self.loss_masks = self._encode_pairs(pairs)

        n = int(0.9 * len(self.input_ids))
        self.train_inputs,  self.val_inputs  = self.input_ids[:n],  self.input_ids[n:]
        self.train_targets, self.val_targets = self.target_ids[:n], self.target_ids[n:]
        self.train_masks,   self.val_masks   = self.loss_masks[:n], self.loss_masks[n:]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_batch(self, split: str):
        inputs  = self.train_inputs  if split == "train" else self.val_inputs
        targets = self.train_targets if split == "train" else self.val_targets
        masks   = self.train_masks   if split == "train" else self.val_masks

        ix = torch.randint(len(inputs), (self.batch_size,))
        return (
            inputs[ix].to(self.device),
            targets[ix].to(self.device),
            masks[ix].to(self.device),
        )

    def encode(self, s: str):
        return [self.ctoi[c] for c in s if c in self.ctoi]

    def decode(self, ids):
        return "".join(self.itoc[i] for i in ids)

    def make_prompt(self, original: str) -> str:
        """Return the instruction prefix for generation-time prompting."""
        return (
            "Instruction:\n"
            "Rewrite this in modern English:\n"
            f"'{original}'\n\n"
            "Response:\n"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(s: str) -> str:
        import re
        s = s.replace('\xa0', ' ')
        s = s.replace('\u2018', "'").replace('\u2019', "'")   # smart single quotes
        s = s.replace('\u201c', '').replace('\u201d', '')     # smart double quotes
        s = s.replace('\u2014', '-').replace('\u2013', '-')   # em/en dash
        s = s.replace('\u2026', '...')                        # ellipsis
        s = s.replace('\xe8', 'e').replace('\xe9', 'e')       # e-accents
        s = s.replace('\xe0', 'a').replace('\xe7', 'c')       # a/c accents
        s = s.replace('\xef', 'i').replace('\xeb', 'e')       # i/e accents
        s = re.sub(r'\[.*?\]', '', s)                         # [annotations]
        s = re.sub(r'@\w+', '', s)                            # @CharacterName tags
        s = s.replace('(', '').replace(')', '')
        return s.strip()

    def _load_pairs(self, csv_path: str):
        import csv
        pairs = []
        with open(csv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                orig  = self._normalize(row['original'])
                trans = self._normalize(row['translation'])
                if not orig or not trans:
                    continue
                if all(c in self.ctoi for c in orig) and all(c in self.ctoi for c in trans):
                    pairs.append((orig, trans))
        return pairs

    def _encode_pairs(self, pairs):
        all_inputs, all_targets, all_masks = [], [], []
        pad_fn = torch.nn.functional.pad

        for orig, trans in pairs:
            full_text = (
                "Instruction:\n"
                "Rewrite this in modern English:\n"
                f"'{orig}'\n\n"
                "Response:\n"
                f"{trans}\n\n"
            )
            prefix_len = len(
                "Instruction:\n"
                "Rewrite this in modern English:\n"
                f"'{orig}'\n\n"
                "Response:\n"
            )

            if not all(c in self.ctoi for c in full_text):
                continue

            ids = [self.ctoi[c] for c in full_text]

            # Truncate to block_size + 1 so input fits in block_size
            if len(ids) > self.block_size + 1:
                ids = ids[:self.block_size + 1]

            seq_len = len(ids)
            if seq_len < 2:
                continue

            input_ids  = torch.tensor(ids[:-1], dtype=torch.long)
            target_ids = torch.tensor(ids[1:],  dtype=torch.long)

            loss_mask = torch.zeros(seq_len - 1, dtype=torch.float)
            resp_token_start = prefix_len - 1
            if resp_token_start < seq_len - 1:
                loss_mask[resp_token_start:] = 1.0

            all_inputs.append(input_ids)
            all_targets.append(target_ids)
            all_masks.append(loss_mask)

        if not all_inputs:
            raise ValueError("No valid pairs encoded. Check that the CSV columns are 'original' and 'translation'.")

        max_len = max(t.shape[0] for t in all_inputs)
        return (
            torch.stack([pad_fn(x, (0, max_len - x.shape[0])) for x in all_inputs]),
            torch.stack([pad_fn(x, (0, max_len - x.shape[0])) for x in all_targets]),
            torch.stack([pad_fn(x, (0, max_len - x.shape[0])) for x in all_masks]),
        )
