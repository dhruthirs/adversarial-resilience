import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchattacks

# You would import your teammate's attack functions and models here
# from attacks import fgsm_attack, pgd_attack 
# from models import load_resnet18, load_mobilenet

class AdversarialPipeline:
    def __init__(self, model_name, dataset_name="CIFAR-10", device="cuda"):
        """
        Initializes the plug-and-play pipeline.
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
    def _load_model(self):
        """
        PLUG-AND-PLAY JUNCTION: Add new SOTA models here later.
        """
        print(f"[*] Loading {self.model_name}...")
        if self.model_name == "ResNet18":
            model = torchvision.models.resnet18(weights=None, num_classes=10)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            model = model.to(self.device)
            model.load_state_dict(torch.load("models/resnet18_cifar10.pth", map_location=self.device))
            model.eval()
            return model
        elif self.model_name == "BasicANN":
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 32 * 32, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
            model = model.to(self.device)
            model.load_state_dict(torch.load("models/basic_ann_cifar10.pth", map_location=self.device))
            model.eval()
            return model
        elif self.model_name == "MobileNetV2":
            return None # Replace with: load_mobilenet().to(self.device)
        elif self.model_name == "DenseNet":
            return None # Replace with: load_densenet().to(self.device)
        # LATER: elif self.model_name == "ViT-Base": return load_vit()
        else:
            raise ValueError(f"Model {self.model_name} not supported yet.")

    def evaluate_clean(self, test_loader):
        """Calculates Baseline Clean Accuracy"""
        print("[*] Evaluating Clean Accuracy...")
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        clean_acc = 100 * correct / total
        return clean_acc

    def evaluate_attack(self, test_loader, attack_type="PGD", epsilon=8/255):
        """
        Runs the attack and calculates Robust Accuracy and ASR.
        """
        print(f"[*] Running {attack_type} Attack with eps={epsilon:.4f}...")
        self.model.eval()
        if attack_type == "FGSM":
            attack = torchattacks.FGSM(self.model, eps=epsilon)
        elif attack_type == "PGD":
            attack = torchattacks.PGD(self.model, eps=epsilon, alpha=2/255, steps=20)
        else:
            raise ValueError(f"Attack type {attack_type} not supported.")

        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            adv_images = attack(images, labels)
            with torch.no_grad():
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        robust_acc = 100 * correct / total
        asr = 100 - robust_acc
        return robust_acc, asr

    def run_full_benchmark(self, test_loader, epsilon=8/255):
        """
        The master function. Runs clean, FGSM, and PGD and returns the dictionary.
        """
        clean_acc = self.evaluate_clean(test_loader)
        _, fgsm_asr = self.evaluate_attack(test_loader, attack_type="FGSM", epsilon=epsilon)
        _, pgd_asr = self.evaluate_attack(test_loader, attack_type="PGD", epsilon=epsilon)
        
        results = {
            "model_name": self.model_name,
            "dataset": self.dataset_name,
            "epsilon": epsilon,
            "clean_accuracy": clean_acc,
            "fgsm_asr": fgsm_asr,
            "pgd_asr": pgd_asr
        }
        return results

# ==========================================
# HOW TO USE THIS SKELETON (The Plug-and-Play)
# ==========================================
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
#    test_set = datasets.ImageFolder(root='cifar10/test', transform=transform)  Not in local folder hence...
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)
    
    # 1. Test ResNet
    pipeline_resnet = AdversarialPipeline(model_name="ResNet18")
    results_resnet = pipeline_resnet.run_full_benchmark(test_loader)
    print(results_resnet)
    
    # 2. Test DenseNet (Just swap the string!)
    pipeline_dense = AdversarialPipeline(model_name="DenseNet")
    results_dense = pipeline_dense.run_full_benchmark(test_loader)
    print(results_dense)