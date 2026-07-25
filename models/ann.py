#ann
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchattacks
import matplotlib.pyplot as plt

# 1. Configuration & GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=False) 

# 3. Define an Artificial Neural Network (ANN / Multi-Layer Perceptron)
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        # MNIST images are 1x28x28 = 784 pixels
        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 output classes for digits 0-9
        )

    def forward(self, x):
        # Flatten the 2D image (Batch, 1, 28, 28) into a 1D vector (Batch, 784)
        x = x.view(-1, 784)
        x = self.network(x)
        return x

model = SimpleANN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the ANN Model
print("\nStarting ANN model training on MNIST using GPU...")
model.train()
for epoch in range(2):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/2 complete.")
print("Training finished!")

# 5. Initialize the Attacks
# (torchattacks automatically handles the forward pass flattening we built inside the ANN)
fgsm_attack = torchattacks.FGSM(model, eps=0.25)
pgd_attack = torchattacks.PGD(model, eps=0.25, alpha=2/255, steps=40)

# 6. Evaluate the Performance
model.eval()
correct_clean = 0
correct_fgsm = 0
correct_pgd = 0
total = 0

saved_clean_img = None
saved_fgsm_img = None
saved_pgd_img = None
saved_clean_pred = None
saved_fgsm_pred = None
saved_pgd_pred = None

print("\nEvaluating Clean vs FGSM vs PGD attacks on the ANN model...")
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
    
    # Save the first sample from the first batch for visualization
    if saved_clean_img is None:
        saved_clean_img = images.cpu().data[0][0]
        saved_fgsm_img = fgsm_images.cpu().data[0][0]
        saved_pgd_img = pgd_images.cpu().data[0][0]
        saved_clean_pred = predicted_clean[0].item()
        saved_fgsm_pred = predicted_fgsm[0].item()
        saved_pgd_pred = predicted_pgd[0].item()

# 7. Print Terminal Scoreboard Report
acc_clean = (correct_clean / total) * 100
acc_fgsm = (correct_fgsm / total) * 100
acc_pgd = (correct_pgd / total) * 100

print("\n" + "="*40)
print("       ANN ATTACK SCOREBOARD       ")
print("="*40)
print(f"ANN Accuracy on CLEAN images: {acc_clean:.2f}%")
print(f"ANN Accuracy under FGSM attack: {acc_fgsm:.2f}% (Drop: {acc_clean - acc_fgsm:.2f}%)")
print(f"ANN Accuracy under PGD attack:  {acc_pgd:.2f}% (Drop: {acc_clean - acc_pgd:.2f}%)")
print("="*40)

# 8. Create and Save Comparative Visual Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(saved_clean_img, cmap='gray')
axes[0].set_title(f"Clean Image\nPrediction: {saved_clean_pred}")
axes[0].axis('off')

axes[1].imshow(saved_fgsm_img, cmap='gray')
axes[1].set_title(f"FGSM Attack (ANN)\nPrediction: {saved_fgsm_pred}")
axes[1].axis('off')

axes[2].imshow(saved_pgd_img, cmap='gray')
axes[2].set_title(f"PGD Attack (ANN)\nPrediction: {saved_pgd_pred}")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('ann_fgsm_vs_pgd_comparison.png')
print("\nComparative visual saved successfully as 'ann_fgsm_vs_pgd_comparison.png'!")