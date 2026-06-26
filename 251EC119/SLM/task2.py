# Load the GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

dataset = load_dataset("roneneldan/TinyStories")

def tokenize(example):
    ids = enc.encode(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}

tokenized = dataset.map(tokenize, remove_columns=["text"])

for split in ["train", "validation"]:
    for ids in tokenized[split]["ids"]:
        print(len(ids))
        break

for split in ["train", "validation"]:

    total_tokens = sum(tokenized[split]["len"])

    filename = f"{split}.bin"

    arr = np.memmap(
        filename,
        dtype=np.uint16,
        mode="w+",
        shape=(total_tokens,)
    )

    idx = 0

    for ids in tokenized[split]["ids"]:
        arr[idx:idx + len(ids)] = ids
        idx += len(ids)

    arr.flush()

print("Saved train.bin and validation.bin")

import os
print(os.listdir())

# Load the binary files
train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("validation.bin", dtype=np.uint16, mode="r")

# Print total number of tokens
print(f"Train tokens: {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}")

# Print first 50 token IDs
first_50 = train_data[:50]

print("\nFirst 50 token IDs:")
print(first_50)

# Decode back into text
decoded_text = enc.decode(first_50.tolist())

print("\nDecoded text:")
print("\nFirst 50 Tokens Decoded.\n")
print(decoded_text)
