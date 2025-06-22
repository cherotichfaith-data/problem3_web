# File: app.py
import streamlit as st
import torch
import torch.nn as nn  # ✅ Required for nn.Module, nn.Embedding, etc.
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Generator class copied from the training script
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        return self.model(x).view(-1, 1, 28, 28)

@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model

def generate_images(digit, model):
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        images = model(z, labels).view(-1, 28, 28)
    return images

# Streamlit UI
st.title("Handwritten Digit Generator")
digit = st.selectbox("Choose a digit (0–9)", list(range(10)))

if st.button("Generate"):
    model = load_model()
    imgs = generate_images(digit, model)
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, img in enumerate(imgs):
        axs[i].imshow(img.numpy(), cmap='gray')
        axs[i].axis("off")
    st.pyplot(fig)
