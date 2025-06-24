
# 🧠 MiniLLM – Tiny Transformer Chatbot from Scratch

This is a personal deep learning project where I built and trained a minimal character-level Transformer language model to generate chatbot responses. The model is decoder-only (like GPT) and trained on a small dialogue dataset using PyTorch.

---

## 💡 What it does

- Implements a decoder-only Transformer (multi-head attention + positional embeddings)
- Tokenizes text character-by-character with a custom tokenizer
- Trains on a simple dialogue corpus between User and Bot
- Generates responses conditioned on user input using temperature sampling

---

## 📈 Project Status

🚧 This project is **still in progress**. Several improvements are planned:

- 📊 Add a **larger and more diverse dataset** for training  
- ⚡ Allow **training on GPU** (currently CPU-only)
- 📦 Add better sampling (e.g., top-k, nucleus sampling)
- 🧠 Improve model depth and regularization
- 📄 Save/load tokenizer alongside model

---

## ▶️ How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```
The model will be saved in `checkpoints/mini_gpt.pth`.

3. Run the chatbot:
```bash
python generate.py
```

You can then interact with the chatbot in your terminal.

---

## 📂 Project Structure

```
mini-llm/
├── dataset.txt        # Training data (User/Bot format)
├── model.py           # Decoder-only Transformer implementation
├── tokenizer.py       # Char-level tokenizer
├── train.py           # Training script
├── generate.py        # Inference script (chat interface)
├── utils.py           # Model saving/loading utilities
└── checkpoints/       # Saved model
```

---

## 📌 Technologies Used

PyTorch, Python 3.10+, TransformerDecoder, Custom Tokenizer

---
