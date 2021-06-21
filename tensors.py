import torch
import numpy as np

# Tensors are specialised data structures that are similar to arrays and matrices.
# Tensors encode the inputs and outputs of a model as well as model parameters.

# Similar to NPs ndarrys, except they can run on GPUs or other hardware accelerators.


# Initialising Tensors

# Directly from Data:
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From a NumPy array:
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor:
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# With random or constant values:
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# Tensor Attributes
# Describe their shape, datatype and the device they are stored.
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Operations on Tensors
# Linear Algebra, arithmetic, matric manipulation (transposing, indexing, slicing), sampling +
# These can run on either the GPU or CPU.
# Tensors are created on CPU by default and can be moved to GPU if available.
# Copying large tensors across can be expensive in time and memory!
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

tensor = torch.ones(4, 4)
print("First Row: ", tensor[0])
print("First Column: ", tensor[:, 0])
print("Last Column: ", tensor[..., -1])
tensor[:,1] = 0
print(tensor)

# Joining Tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic
# Matrix Multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
# All 3 are equal
print(y1)
# print(y2)
# print(y3)

# Element-wide product
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)

# Single element tensors
# item() converts to Python numerical value.
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations
# Suffixed with _
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# Bridge with Numpy
# Tensors on the CPU and NumPy arrays can share their underlying memory locations.
# Changing one will change the other

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")