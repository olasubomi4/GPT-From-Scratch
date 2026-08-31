import glob
import os
import torch
from data_loader import DataLoader
from gpt_modules.model import BigramLanguageModel
from config import batch_size, block_size, n_embed, n_head, n_layer, dropout, device
from sft_data_loader import ModernizationSFTDataLoader, CorpusSFTDataLoader
from config import sft_source_len


def find_latest_weights(prefix: str):
    candidates = glob.glob(os.path.join(os.path.dirname(__file__), f"{prefix}_*.pth"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def main():
    with open("./data/input.txt", "r") as f:
        text = f.read()

    dl = DataLoader(text, batch_size, block_size, device)
    model = BigramLanguageModel(
        vocab_size=dl.vocab_size,
        n_embed=n_embed,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device,
        profile=None
    ).to(device)

    weights = find_latest_weights("sft_weights")
    if weights:
        print(f"Loading weights: {weights}")
        model.load_state_dict(torch.load(weights, map_location=device))
    else:
        print("No SFT weights found — using random init")

    model.eval()

    # Build the data loader just to get the make_prompt helper
    modern_dl = CorpusSFTDataLoader(
        csv_path="/Users/olasubomiodekunle/Downloads/corpus.csv",
        pretrain_text=text,
        batch_size=1,
        block_size=block_size,
        device=device,
    )

    # test_sources = [
    #     "All the world's a stage.",
    #     "To be, or not to be, that is the question.",
    #     "What's in a name?",
    #     "Some are born great, some achieve greatness, and some have greatness thrust upon them.",
    #     "The course of true love never did run smooth.",
    # ]

    test_sources = [
        "Thou art more lovely and more temperate.",
        "Thou hast my heart.",
        "I know not what to say.",
        "Dost thou hear me?",
        "What sayest thou?",
        "Whither shall I go?",
        "Wherefore dost thou weep?",
        "Hath he gone?",
        "Methinks the lady doth protest too much.",
        "Pray tell me, what is thy name?",
        "I beseech thee, listen to me.",
        "Get thee hence.",
        "Come hither.",
        "Look upon me.",
        "Speak not of it.",
        "I shall not forget thee.",
        "What manner of man is this?",
        "How fares thy father?",
        "Would that I were young again.",
        "'Tis better to have loved and lost.",
        "The night is dark and full of dreams.",
    ]

    # EOS stop sequence: two consecutive newlines
    newline_id = dl.ctoi.get('\n')
    stop_token_ids = [newline_id, newline_id] if newline_id is not None else None

    for source in test_sources:
        prompt = modern_dl.make_prompt(source)
        # Skip chars not in vocab
        prompt = "".join(c for c in prompt if c in dl.ctoi)
        prompt_ids = torch.tensor([dl.encode(prompt)], dtype=torch.long, device=device)

        generated = model.generate(
            prompt_ids,
            max_new_tokens=120,
            use_cache=False,
            temperature=0.0,
            top_k=100,
            repetition_penalty=0,
            stop_token_ids=stop_token_ids,
        )

        full = dl.decode(generated[0].tolist())
        # Print only the response portion, strip trailing \n\n
        response_marker = "Response:\n"
        if response_marker in full:
            response = full.split(response_marker)[-1]
        else:
            response = full[len(prompt):]

        print(f"Source:   {source}")
        print(f"Response: {response.strip()}")
        print()


if __name__ == "__main__":
    main()
