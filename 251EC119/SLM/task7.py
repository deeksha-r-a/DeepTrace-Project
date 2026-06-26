# Load best checkpoint
model.load_state_dict(torch.load("best_model.pt", map_location=device))
# model.to(device)
model.eval()

# validation loss and perplexity
@torch.no_grad()
def evaluate_val_loss(model, val_loader, eval_iters=100):
    losses = []

    for _ in range(eval_iters):
        xb, yb = next(iter(val_loader))
        # xb, yb = xb.to(device), yb.to(device)

        _, loss = model(xb, yb)
        losses.append(loss.item())

    return sum(losses) / len(losses)


val_loss = evaluate_val_loss(model, val_loader)

perplexity = math.exp(val_loss)

print(f"Validation Loss: {val_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")

# Text generation
def generate_text(model, prompt, max_new_tokens=100):
    model.eval()

    tokens = tokenize(prompt)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(tokens)

            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat((tokens, next_token), dim=1)

    return enc.decode(tokens[0].tolist())

prompts = [
    "Once upon a time",
    "The little boy",
    "In the forest",
    "The magic tree"
]

for p in prompts:
    print("=" * 50)
    print("Prompt:", p)
    print(generate_text(model, p))
    print()


