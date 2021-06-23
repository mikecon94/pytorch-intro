import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Modules in PyTorch subclass the nn.Module.
# An NN itself is a module cconsisting of other modules (layers).

# Get device for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Define NN by subclassing nn.Module
# nn.Module subclasses implement the operations on input data in the forward method
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted Class: {y_pred}")

# Model Layers Walkthrough
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten - converts each 2D 28x28 image into a single 784 item array.
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear - Applies a linear transformation to the input using the weights and biases.
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU - Non-linear activation.
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential - Ordered container of modules. Data is passed through all the modules in the same order as defined.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax - The last layer of the nn returns logits-values in [-inf, inf]. Softmax scales to [0, 1] giving a probability.
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# Model Parameters - Many layers inside a NN are parameterized - have associated weights & biases that are optimized during training.
# Subclassing nn.Module automatically tracks all fields defined inside your model object and makes all parameters accessible using the model's parameters() or named_parameters() method.
print("Model Structure: ", model, "\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}")
