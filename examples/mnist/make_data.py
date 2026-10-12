"""MNIST as learnc data files: px (examples x 784 floats in 0..1) and output (the digit)."""
import os
import torch
from torchvision import datasets

here = os.path.dirname(os.path.abspath(__file__))
for split, train in (("train", True), ("test", False)):
    d = datasets.MNIST(os.path.join(here, "..", "..", "data"), train=train, download=True)
    torch.save({"px": d.data.reshape(len(d.data), 784).float() / 255, "output": d.targets}, os.path.join(here, f"{split}.pt"))
print("wrote train.pt and test.pt")
