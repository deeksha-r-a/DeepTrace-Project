# Loss estimation function
@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters=50):
    out = {}
    model.eval()

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = torch.zeros(eval_iters)

        loader_iter = iter(loader)

        for k in range(eval_iters):
            try:
                X, Y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                X, Y = next(loader_iter)

            #X, Y = X.to(device), Y.to(device)

            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out

    # Training loop
def train_model(model, optimizer, train_loader, val_loader,
                max_iters=2000, eval_interval=50):

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for iteration in range(max_iters):

        train_iter = iter(train_loader)
        xb, yb = next(train_iter)
        #xb, yb = xb.to(device), yb.to(device)

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration % eval_interval == 0 or iteration == max_iters - 1:

            losses = estimate_loss(model, train_loader, val_loader)

            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

            print(f"Iter {iteration}: "
                  f"train {losses['train']:.4f}, "
                  f"val {losses['val']:.4f}")

            print("Current val:", losses["val"], "Best:", best_val_loss)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), "best_model.pt")
                print("Saved best_model.pt")

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

    return model

# Run training
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y
block_size = 128
train_dataset = TokenDataset(train_data, block_size)
val_dataset = TokenDataset(val_data, block_size)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
model = train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    max_iters=2000,
    eval_interval=50
)
