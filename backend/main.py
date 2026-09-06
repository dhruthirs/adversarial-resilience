from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
import os


app = FastAPI(title="Adversarial Resilience API")


# -------------------------
# CORS
# -------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------
# ANN Model
# -------------------------

class SimpleANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):

        x = x.view(x.size(0), -1)

        return self.net(x)


# -------------------------
# CNN Model
# -------------------------

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.classifier(x)

        return x


# -------------------------
# Model Paths
# -------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ANN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "weights",
    "ann.pth"
)

CNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "weights",
    "cnn.pth"
)


# -------------------------
# Load ANN
# -------------------------

ann_model = SimpleANN().to(device)

ann_model.load_state_dict(
    torch.load(
        ANN_MODEL_PATH,
        map_location=device
    )
)

ann_model.eval()

print("ANN model loaded successfully.")


# -------------------------
# Load CNN
# -------------------------

cnn_model = SimpleCNN().to(device)

cnn_model.load_state_dict(
    torch.load(
        CNN_MODEL_PATH,
        map_location=device
    )
)

cnn_model.eval()

print("CNN model loaded successfully.")


# -------------------------
# Image Preprocessing
# -------------------------

transform = transforms.Compose([
    transforms.Grayscale(
        num_output_channels=1
    ),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


# -------------------------
# FGSM Attack
# -------------------------

def fgsm_attack(
    model,
    image,
    label,
    epsilon
):

    image = image.clone().detach()

    image.requires_grad = True

    output = model(image)

    loss = nn.CrossEntropyLoss()(
        output,
        label
    )

    model.zero_grad()

    loss.backward()

    gradient = image.grad.data

    adversarial_image = (
        image
        + epsilon * gradient.sign()
    )

    adversarial_image = torch.clamp(
        adversarial_image,
        0,
        1
    )

    return adversarial_image.detach()


# -------------------------
# PGD Attack
# -------------------------

def pgd_attack(
    model,
    image,
    label,
    epsilon,
    alpha,
    steps
):

    original_image = image.clone().detach()

    adversarial_image = image.clone().detach()

    for _ in range(steps):

        adversarial_image.requires_grad = True

        output = model(adversarial_image)

        loss = nn.CrossEntropyLoss()(
            output,
            label
        )

        model.zero_grad()

        loss.backward()

        gradient = adversarial_image.grad.data

        adversarial_image = (
            adversarial_image
            + alpha * gradient.sign()
        )

        perturbation = (
            adversarial_image
            - original_image
        )

        perturbation = torch.clamp(
            perturbation,
            -epsilon,
            epsilon
        )

        adversarial_image = (
            original_image
            + perturbation
        )

        adversarial_image = torch.clamp(
            adversarial_image,
            0,
            1
        ).detach()

    return adversarial_image


# -------------------------
# Root
# -------------------------

@app.get("/")
def root():

    return {
        "message":
        "Adversarial Resilience API is running!"
    }


# -------------------------
# Upload
# -------------------------

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    model: str = Form(...),
    attack: str = Form(...)
):

    image_data = await file.read()

    image = Image.open(
        io.BytesIO(image_data)
    )

    return {
        "filename": file.filename,
        "width": image.width,
        "height": image.height,
        "model": model,
        "attack": attack,
        "message":
        "Image, model, and attack received successfully!"
    }


# -------------------------
# Predict + Attack
# -------------------------

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("ANN"),
    attack: str = Form("FGSM")
):

    # -------------------------
    # Select Model
    # -------------------------

    model_name = model_name.upper()

    if model_name == "ANN":

        selected_model = ann_model

    elif model_name == "CNN":

        selected_model = cnn_model

    else:

        return {
            "message":
            f"Unsupported model: {model_name}"
        }


    # -------------------------
    # Read Image
    # -------------------------

    image_data = await file.read()

    image = Image.open(
        io.BytesIO(image_data)
    ).convert("L")


    # -------------------------
    # Preprocess
    # -------------------------

    image_tensor = transform(image)

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )


    # -------------------------
    # Clean Prediction
    # -------------------------

    with torch.no_grad():

        clean_output = selected_model(
            image_tensor
        )

    clean_probabilities = torch.softmax(
        clean_output,
        dim=1
    )

    clean_prediction = (
        clean_output
        .argmax(dim=1)
        .item()
    )

    clean_confidence = (
        clean_probabilities[
            0,
            clean_prediction
        ].item()
    )


    # -------------------------
    # Attack Label
    # -------------------------

    # The true label of an uploaded
    # image is unknown, so we use the
    # clean prediction as the label.

    attack_label = torch.tensor(
        [clean_prediction],
        device=device
    )


    # -------------------------
    # Attack
    # -------------------------

    if attack.upper() == "FGSM":

        adversarial_image = fgsm_attack(
            selected_model,
            image_tensor,
            attack_label,
            epsilon=0.25
        )

    elif attack.upper() == "PGD":

        adversarial_image = pgd_attack(
            selected_model,
            image_tensor,
            attack_label,
            epsilon=0.25,
            alpha=2 / 255,
            steps=40
        )

    else:

        return {
            "message":
            f"Unsupported attack: {attack}"
        }


    # -------------------------
    # Adversarial Prediction
    # -------------------------

    with torch.no_grad():

        adversarial_output = selected_model(
            adversarial_image
        )

    adversarial_probabilities = torch.softmax(
        adversarial_output,
        dim=1
    )

    adversarial_prediction = (
        adversarial_output
        .argmax(dim=1)
        .item()
    )

    adversarial_confidence = (
        adversarial_probabilities[
            0,
            adversarial_prediction
        ].item()
    )


    # -------------------------
    # Attack Status
    # -------------------------

    attack_success = (
        adversarial_prediction
        != clean_prediction
    )


    # -------------------------
    # Response
    # -------------------------

    return {
        "model": model_name,
        "attack": attack.upper(),

        "clean_prediction":
            clean_prediction,

        "clean_confidence":
            round(clean_confidence, 4),

        "adversarial_prediction":
            adversarial_prediction,

        "adversarial_confidence":
            round(adversarial_confidence, 4),

        "attack_success":
            attack_success,

        "message":
            "Analysis completed successfully."
    }