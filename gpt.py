import os
from datetime import datetime
import torch
from dotenv import load_dotenv

from config import batch_size, block_size, max_iters, eval_interval, learning_rate, device, eval_iters, n_embed, n_head, n_layer, dropout
from data_loader import DataLoader
from gpt_modules.model import BigramLanguageModel
from gpt_modules.utils import TimingProfiler

@torch.no_grad()
def estimate_loss(model, data_loader):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = data_loader.get_batch(split)
            logits, loss = model(x, y, use_cache=False)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    torch.manual_seed(1337)
    
    # read file
    data_path = "./data/input.txt"
    if not os.path.exists(data_path):
        # Fallback for testing if data/input.txt doesn't exist in the current environment
        print(f"Warning: {data_path} not found.")
        # Try to find it or create a dummy one if we are in a restricted environment
        if not os.path.exists("data"):
            os.makedirs("data")
        with open(data_path, "w") as f:
            f.write("ABC") # Dummy data

    with open(data_path, "r") as f:
        text = f.read()

    # Data loading
    data_loader = DataLoader(text, batch_size, block_size, device)
    vocab_size = data_loader.vocab_size
    
    # Profiler
    profile = TimingProfiler()
    
    # Model
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        n_embed=n_embed,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device,
        profile=profile
    )
    m = model.to(device)


    # Check for pre-trained weights
    weights_path = "/Users/olasubomiodekunle/PycharmProjects/GPT-From-Scratch/model_weights_2026-07-23 22:36:09.437281.pth"
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        is_trained = True
    else:
        print("No pre-trained weights found, starting training...")
        is_trained = False

    if not is_trained:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        print(f"Training starts now on device: {device}")

        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss(model, data_loader)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = data_loader.get_batch("train")
            logits, loss = model(xb, yb, use_cache=False)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(model.state_dict(), f"model_weights_{timestamp}.pth")

    # Generation
    load_dotenv()
    use_kv_cache = os.getenv("USE_KV_CACHE", "true").lower() == "true"
    print(f"KV Cache enabled: {use_kv_cache}")

    m.eval()
    torch.random.manual_seed(42)
    print("######################################################################")
    print(f"Generating:")
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    start_time = datetime.now()
    result_ids = m.generate(context, max_new_tokens=500, use_cache=use_kv_cache)[0].tolist()
    end_time = datetime.now()
    
    print(data_loader.decode(result_ids))
    profile.summary()
    print(f"Time taken to generate: {end_time - start_time}")

if __name__ == "__main__":
    main()
