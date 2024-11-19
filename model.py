# model.py
import torch
import torch.nn as nn


class CharTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(CharTransformerModel, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout)

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # Устанавливаем batch_first=True
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.src_mask = None

    def generate_square_subsequent_mask(self, sz):
        if self.src_mask is None or self.src_mask.size(0) != sz:
            mask = torch.triu(torch.ones(sz, sz), 1)
            self.src_mask = mask.masked_fill(mask == 1, float('-inf')).to(next(self.parameters()).device)
        return self.src_mask

    def forward(self, src):
        """
        src shape: (batch_size, seq_length)
        """
        seq_length = src.size(1)
        embed = self.embedding(src) * torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32, device=src.device))
        embed = self.positional_encoding(embed)  # Shape: (batch_size, seq_length, embed_size)

        mask = self.generate_square_subsequent_mask(seq_length)
        # Transformer expects src and tgt; для генерации используем src=embed и tgt=embed
        output = self.transformer(src=embed, tgt=embed, src_mask=mask, tgt_mask=mask)
        logits = self.fc_out(output)  # Shape: (batch_size, seq_length, vocab_size)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_size, device='cpu')
        position = torch.arange(0, max_len, dtype=torch.float32, device='cpu').unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2, dtype=torch.float32, device='cpu') * (
                    -torch.log(torch.tensor(10000.0)) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_size % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_size)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)