from task2 import train_loader, optimizer
from task3 import Net

model = Net()

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()

    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()                   # Step 1: Clear old gradients
        output = model(data)                     # Step 2: Forward pass
        loss = criterion(output, target)         # Step 3: Compute loss
        loss.backward()                          # Step 4: Backpropagation
        optimizer.step()                         # Step 5: Update weights

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} "
                f"[{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = running_loss / len(train_loader)
    return avg_loss