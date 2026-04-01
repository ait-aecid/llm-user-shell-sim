import torch
import torch_directml

# Create DirectML device (this maps to Intel iGPU)
dml = torch_directml.device()

# Create tensors explicitly on the iGPU
x = torch.randn(64, 64, device=dml)
y = torch.randn(64, 64, device=dml)

# GPU computation
z = x @ y

print("Device:", z.device)