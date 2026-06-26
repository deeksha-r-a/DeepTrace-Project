# Assuming these are defined elsewhere in your script
batch_size = 16
block_size = 256
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    # Select the correct data file
    data = train_data if split == 'train' else val_data

    # Generate random starting indices for the batch
    # We subtract block_size because y is shifted by 1 (needs i + block_size + 1)
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stack the sequences into tensors
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    # Move tensors to the designated device (CPU or CUDA)
    #x, y = x.to(device), y.to(device)

    return x, y

# Test the implementation
x, y = get_batch('train')
print(f"x shape: {x.shape}") #(batch_size, block_size)
print(f"y shape: {y.shape}") #(batch_size, block_size)

# Verify shifting: y[0, 0] should equal x[0, 1]
print(f"Verification: {torch.equal(x[0, 1:], y[0, :-1])}")
