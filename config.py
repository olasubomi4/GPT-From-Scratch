import torch

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

# SFT hyperparameters
sft_max_iters = 2000
sft_learning_rate = 1e-4
sft_eval_interval = 200
sft_eval_iters = 50
sft_source_len = 80        # chars per archaic source chunk
sft_early_stop_patience = 3   # eval intervals with no val improvement before stopping
sft_overfit_gap = 0.15        # val_loss - train_loss gap that also triggers early stop