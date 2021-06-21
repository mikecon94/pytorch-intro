from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# All TorchVision datasets have two parameters
# transform - Modify the features
# traget_transform - Modify the labels
# They accept callables containing the logic.

# FashionMNIST features are in PIL Image format and labels are integers.
# We need features as normalised tensors and labels as one-hot encoded tensors for training.
# To do this we use ToTensor and Lambda.

ds = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# ToTensor()
# Converts a PIL image or NumPy ndarray into a FloatTensor and scales the image's pixel intensity values in the range
# [0., 1.]

# Lambda Transforms
# Apply any user-defined lambda function.
# Here, we define a function to turn the integer into a one-hot encoded tensor.
# First creates a zero tensor of size 10 (number of labels in dataset)
# Then calls scatter_() which assigns a value=1 on the index as given by the label y.