import torch

class DataLoader:
    def __init__(self, text, batch_size, block_size, device):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        
        self.ctoi = {c: id for id, c in enumerate(self.chars)}
        self.itoc = {id: c for id, c in enumerate(self.chars)}
        
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):
        return [self.ctoi[c] for c in s]

    def decode(self, ids):
        return "".join([self.itoc[id] for id in ids])

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
