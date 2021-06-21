import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Datasets & DataLoaders

# Dataset stores samples and corresponding labels
# DataLoader wraps an iterable around the Dataset

# PyTorch libraries include pre-loaded datasets like FashionMNIST

# Loading FashionMNIST
# root - path the train/test data is stored
# train - specificaies training or test dataset
# download - Downloads the data from the internet if it's not available at root
# transform / target_transform - specify the feature and label transformations

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

# Iterating and visualising the dataset
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# Custom Dataset for files - Must implement __init, __len__, __getitem__

# DataLoaders
# Dataset retrieves features and labels one sample at a time.
# When training, we typically want to pass samples in minibatches, reshuffle the data at every epoch
# and use multiprocessing to speed up data retrieval.
# DataLoader abstracts this complexity for us.
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
print(f"Squeezed Batch shape: {img.size()}")
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")