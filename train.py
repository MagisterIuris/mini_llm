import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizer import CharTokenizer
from model import MiniTransformerLM  
from utils import save_model

block_size = 128
batch_size = 64
epochs = 20
lr = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer()
vocab_size = tokenizer.vocab_size
encoded_text = tokenizer.encode(text)

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+block_size+1], dtype=torch.long)
        return x, y

dataset = CharDataset(encoded_text, block_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = MiniTransformerLM(vocab_size=vocab_size, block_size=block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    total_loss = 0
    model.train()
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        B, T, C = logits.shape
        loss = loss_fn(logits.view(B*T, C), yb.view(B*T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"🧠 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

save_model(model)