# Load checkpoint
model.load_state_dict(torch.load("best_model.pt", map_location=device))
#model.to(device)
model.eval()

# Tokenizer helpers
enc = tiktoken.get_encoding("gpt2")

def encode(text):
    return torch.tensor(enc.encode(text), dtype=torch.long).unsqueeze(0)

def decode(ids):
    return enc.decode(ids)

# Generate function with temperature
def generate(model, prompt, max_new_tokens=100, temperature=1.0):
    model.eval()

    idx = encode(prompt)

    for _ in range(max_new_tokens):
        logits, _ = model(idx)

        logits = logits[:, -1, :] / temperature   # apply temperature
        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return decode(idx[0].tolist())

# Run inference
prompt = "Once upon a time"

output = generate(model, prompt, max_new_tokens=100, temperature=1.0)

print("OUTPUT(temp=1.0)")
print(output)

# Second temperature
output_low = generate(model, prompt, max_new_tokens=100, temperature=0.7)

print("\nOUTPUT(temp=0.7)")
print(output_low)

#Temperature:
#Temperature controls how random the model’s output is during text generation. 
#A lower temperature (e.g., 0.7) makes the output more focused, repetitive, and grammatically safe, while a medium value (1.0) gives a balance of creativity and coherence. 
#A higher temperature (e.g., 1.3) makes the text more diverse and creative but can also make it less logical or more chaotic.
