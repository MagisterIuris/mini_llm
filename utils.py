import torch
import os

def save_model(model, path="checkpoints/mini_gpt.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Modèle sauvegardé dans {path}")

def load_model(model_class, tokenizer, path="checkpoints/mini_gpt.pth", block_size=128):
    model = model_class(vocab_size=tokenizer.vocab_size, block_size=block_size)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    print(f"Modèle chargé depuis {path}")
    return model


def print_config():
    print("\n--> CONFIGURATION:")
    print("- Block size: 128")
    print("- Batch size: 64")
    print("- Epochs: 20")
    print("- Learning rate: 3e-4")
    print("- Optimizer: AdamW")
    print("- Device:", "cuda" if torch.cuda.is_available() else "cpu")
