# kan
import os
# Set before importing torch to fix CUDA allocator memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchattacks
import gc

# 1. Configuration & GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear any cached VRAM fragments
gc.collect()
torch.cuda.empty_cache()

# Import KAN layer
from efficient_kan import KAN

# 2. Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Batch size configured for stable B-spline autograd backprop memory usage
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 3. Define the Kolmogorov-Arnold Network (KAN)
class MNISTKAN(nn.Module):
    def __init__(self):
        super(MNISTKAN, self).__init__()
        # grid_size=3 keeps B-spline evaluation lightweight
        self.kan = KAN(layers_hidden=[784, 64, 10], grid_size=3, spline_order=3)

    def forward(self, x):
        x = x.reshape(x.size(0), -1).contiguous()
        x = self.kan(x)
        return x

model = MNISTKAN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, foreach=False)

# Mixed Precision Scaler to optimize memory usage
scaler = torch.amp.GradScaler('cuda')

# 4. Train the KAN Model
print("\nStarting KAN model training on MNIST using GPU...")
model.train()
for epoch in range(2):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Cast forward pass & spline math to FP16
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/2 complete. Loss: {running_loss/len(train_loader):.4f}")
    torch.cuda.empty_cache()

print("Training finished!")

# 5. Initialize the Attacks
fgsm_attack = torchattacks.FGSM(model, eps=0.25)
pgd_attack = torchattacks.PGD(model, eps=0.25, alpha=2/255, steps=40)

# 6. Evaluate the Performance
model.eval()

# Freeze model parameters during attack generation
for param in model.parameters():
    param.requires_grad = False

correct_clean = 0
correct_fgsm = 0
correct_pgd = 0
total = 0

print("\nEvaluating Clean vs FGSM vs PGD attacks on the KAN model...")
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    total += labels.size(0)
    
    # 6a. Clean Baseline
    outputs = model(images)
    _, predicted_clean = torch.max(outputs.data, 1)
    correct_clean += (predicted_clean == labels).sum().item()
    
    # 6b. FGSM Evaluation
    fgsm_images = fgsm_attack(images, labels)
    fgsm_outputs = model(fgsm_images)
    _, predicted_fgsm = torch.max(fgsm_outputs.data, 1)
    correct_fgsm += (predicted_fgsm == labels).sum().item()
    
    # 6c. PGD Evaluation
    pgd_images = pgd_attack(images, labels)
    pgd_outputs = model(pgd_images)
    _, predicted_pgd = torch.max(pgd_outputs.data, 1)
    correct_pgd += (predicted_pgd == labels).sum().item()

    torch.cuda.empty_cache()

# 7. Print Terminal Scoreboard Report
acc_clean = (correct_clean / total) * 100
acc_fgsm = (correct_fgsm / total) * 100
acc_pgd = (correct_pgd / total) * 100

print("\n" + "="*40)
print("       KAN ATTACK SCOREBOARD        ")
print("="*40)
print(f"KAN Accuracy on CLEAN images: {acc_clean:.2f}%")
print(f"KAN Accuracy under FGSM attack: {acc_fgsm:.2f}% (Drop: {acc_clean - acc_fgsm:.2f}%)")
print(f"KAN Accuracy under PGD attack:  {acc_pgd:.2f}% (Drop: {acc_clean - acc_pgd:.2f}%)")
print("="*40)



