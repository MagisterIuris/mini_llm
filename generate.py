import torch
from model import MiniTransformerLM
from tokenizer import CharTokenizer
from utils import load_model

tokenizer = CharTokenizer()

model = load_model(MiniTransformerLM, tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt, max_tokens=50):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_tokens):
        if input_tensor.size(1) > model.block_size:
            input_tensor = input_tensor[:, -model.block_size:]

        with torch.no_grad():
            logits = model(input_tensor)
            temperature = 0.7  
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)


        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

    output_ids = input_tensor.squeeze().tolist()
    return tokenizer.decode(output_ids)

if __name__ == "__main__":
    print("📝 Mini GPT – Générateur de texte\nTape 'exit' pour quitter.")
    while True:
        prompt = input("👤 Vous: ")
        if prompt.lower() in {"exit", "quit"}:
            break
        response = generate_text(prompt)
        print(f"🤖 MiniGPT: {response}\n")
