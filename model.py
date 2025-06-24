import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, n_heads=8, n_layers=4, block_size=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(block_size, emb_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=n_heads)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.ln = nn.LayerNorm(emb_dim)
        self.fc_out = nn.Linear(emb_dim, vocab_size)
        self.block_size = block_size

    def forward(self, x):
        B, T = x.shape
        token_emb = self.token_embedding(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x_emb = token_emb + pos_emb
        x_emb = x_emb.transpose(0, 1) 

        tgt_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        decoded = self.transformer(x_emb, memory=torch.zeros_like(x_emb), tgt_mask=tgt_mask)

        decoded = decoded.transpose(0, 1)  
        decoded = self.ln(decoded)
        logits = self.fc_out(decoded)
        return logits
