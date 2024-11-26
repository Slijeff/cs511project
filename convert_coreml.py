import coremltools as ct
import torch
import torchvision

# Define a PyTorch model
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# Example input
example_input = torch.randn(8, 3, 224, 224)

# Convert to Core ML
traced_model = torch.jit.trace(model, example_input)
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    # compute_units=ct.ComputeUnit.CPU_AND_NE
)
coreml_model.save("ResNet18.mlpackage")

