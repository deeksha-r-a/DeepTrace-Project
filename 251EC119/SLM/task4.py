# Define GPTConfig
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        # Causal mask for the attention mechanism
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)

        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x

# Implement the GPT Class
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # positional embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # transformer blocks
            ln_f = nn.LayerNorm(config.n_embd),
            # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Implement forward() method logic
        tok_emb = self.transformer.wte(idx)
        # token embeddings (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # position embeddings (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # (b, t, vocab_size)
        return logits

# Checkpoint Verification
config = GPTConfig(vocab_size=100, block_size=32, n_layer=4, n_head=4, n_embd=128)
model = GPT(config)

# Dummy input tensor (batch_size=4, block_size=32)
dummy_input = torch.zeros((4, 32), dtype=torch.long)
logits = model(dummy_input)

print(f"Output shape: {logits.shape}") # Should be (4, 32, 100)
print(f"Total parameter count: {sum(p.numel() for p in model.parameters()):,}")
