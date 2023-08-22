import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import device, nn, optim
from PIL import Image

model = torch.load('./recognition-model.pt')
model.eval()

def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

def image_loader(img_path):
    loader = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
    ])
    
    image = Image.open(img_path)
    image = loader(image).float()
    return image

img = image_loader('./zero.png')

with torch.no_grad():
    logps = model(img.view(1, -1))

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Dígito previsto =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)