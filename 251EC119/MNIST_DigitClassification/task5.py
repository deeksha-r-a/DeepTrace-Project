import torch
from task2 import test_loader
from task3 import Net

model = Net()


def test(model, device, test_loader, criterion):

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)

    accuracy = 100 * correct / len(test_loader.dataset)

    return avg_loss, accuracy
