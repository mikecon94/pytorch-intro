import torch

# During training the most frequently used algorithm is back propagation.
# During this, parameters are adjusted according to the gradient of the loss function with respect to the given parameter.
# PyTorch has a differentiation engine built-in - torch.autograd.
# Automatic computation of gradient for any computational graph.

# The following code is represented with computational-graph-autograd.png
x = torch.ones(5) # Input Tensor
y = torch.zeros(3) # Expected Output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# w & b are parameters which we need to optimize. Thus, we need to be able to compute the gradients of loss
# with respect to those variables. So we set the requires_grad property of those tensors.
# This can be set later than creation by using x.requires_grad_(True)

# A function applied to tensors to construct computational graph is an object of class Function.
# The object knows how to compute the function in the forward direction and how to compute the derivative.
# A reference to the back prop function is stored in grad_fn property of a tensor.
print("Gradient function for z = ", z.grad_fn)
print("Gradient function for loss = ", loss.grad_fn)

# Computing Gradients
# To optimize weights of parameters we need to compute the derivatives of our loss function with respect to parameters.
# To compute the derivatives we call loss.backward() and retrieve the values.
loss.backward()
print(w.grad)
print(b.grad)
# Grad properties are only available on properties which have requires_grad set to true.
# Backward can only be performed once on a given graph.

# Disabling Gradient Tracking
# Default - All tensors with requires_grad=True are tracking computational history and support gradient computation.
# However, there are cases when we do not need to do that. E.g.
#   When we have trained hte model and just want to apply it to some input data.
#   Mark some parameters as frozen. Useful for finetuning a pretrained network.
# We can stop tracking by using the torch.no_grad() block.
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# We can also use the detach() method
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

# More Info:
# https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#more-on-computational-graphs
# The DAGraph is recreated from scratch; after each .backward() call

# The Jacobian Product can also be calculated instead of the actual gradient
# Call backward with v as an argument.
# v should be the same size as original tensor with respect to which we want to compute the product.
inp = torch.eye(5, requires_grad=True)
out  = (inp + 1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
# Notice the output is different despite the same input being provided.
# This is because pytorch accumulates the gradients.
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)
