import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from task2 import test_dataset
from task3 import Net
from task5 import avg_loss, accuracy

model = Net()

torch.save(model.state_dict(), "mnist_cnn.pt")
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

with torch.no_grad():
    image, label = test_dataset[0]
    image = image.unsqueeze(0).to(torch.device)
    output = model(image)
    prediction = output.argmax(dim=1)

    print("Predicted:", prediction.item())
    print("Actual:", label)



plt.plot(avg_loss, label="Training Loss")
plt.plot(accuracy, label="Test Accuracy")

plt.xlabel("Epoch")
plt.legend()
plt.show()