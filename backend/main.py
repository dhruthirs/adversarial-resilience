from PIL import Image
import uuid
import io
import os

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from torchvision import transforms

from vit_pytorch import ViT
from efficient_kan import KAN

from backend.gan_attack import (
    AttackGenerator,
    generate_adversarial_image
)


# ============================================================
# APP
# ============================================================

app = FastAPI(title="Adversarial Resilience API")


# ============================================================
# CORS
# ============================================================

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


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# ANN MODEL
# ============================================================

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


# ============================================================
# CNN MODEL
# ============================================================

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


# ============================================================
# LSTM MODEL
# ============================================================

class SimpleLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=28,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(
            128,
            10
        )

    def forward(self, x):

        # (B, 1, 28, 28)
        x = x.squeeze(1)

        # (B, 28, 28)
        output, _ = self.lstm(x)

        # Last timestep
        x = output[:, -1, :]

        return self.fc(x)


# ============================================================
# BiLSTM MODEL
# ============================================================

class SimpleBiLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=28,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            256,
            10
        )

    def forward(self, x):

        # (B, 1, 28, 28)
        x = x.squeeze(1)

        # (B, 28, 28)
        output, _ = self.lstm(x)

        # Last timestep
        x = output[:, -1, :]

        return self.fc(x)


# ============================================================
# KAN MODEL
# ============================================================

class MNISTKAN(nn.Module):

    def __init__(self):
        super().__init__()

        self.kan = KAN(
            layers_hidden=[
                784,
                64,
                10
            ],
            grid_size=3,
            spline_order=3
        )

    def forward(self, x):

        x = x.view(
            x.size(0),
            -1
        )

        return self.kan(x)


# ============================================================
# ViT MODEL
# ============================================================

class SimpleViT(ViT):

    def __init__(self):

        super().__init__(
            image_size=28,
            patch_size=7,
            num_classes=10,
            dim=64,
            depth=4,
            heads=4,
            mlp_dim=128,
            channels=1
        )


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

WEIGHTS_DIR = os.path.join(
    BASE_DIR,
    "weights"
)

ADVERSARIAL_DIR = os.path.join(
    BASE_DIR,
    "adversarial_outputs"
)

os.makedirs(
    ADVERSARIAL_DIR,
    exist_ok=True
)


# ============================================================
# STATIC ADVERSARIAL OUTPUT DIRECTORY
# ============================================================

app.mount(
    "/adversarial_outputs",
    StaticFiles(
        directory=ADVERSARIAL_DIR
    ),
    name="adversarial_outputs"
)


# ============================================================
# WEIGHT PATHS
# ============================================================

ANN_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "ann.pth"
)

CNN_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "cnn.pth"
)

LSTM_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "lstm.pth"
)

BILSTM_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "bilstm.pth"
)

KAN_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "kan.pth"
)

VIT_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "vit.pth"
)


# ============================================================
# LOAD MODEL HELPER
# ============================================================

def load_checkpoint(
    model,
    model_path
):

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    # Handle checkpoints that contain
    # state_dict / model_state_dict
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(
        checkpoint
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# LOAD ANN
# ============================================================

ann_model = load_checkpoint(
    SimpleANN(),
    ANN_MODEL_PATH
)

print("ANN model loaded successfully.")


# ============================================================
# LOAD CNN
# ============================================================

cnn_model = load_checkpoint(
    SimpleCNN(),
    CNN_MODEL_PATH
)

print("CNN model loaded successfully.")


# ============================================================
# LOAD LSTM
# ============================================================

lstm_model = load_checkpoint(
    SimpleLSTM(),
    LSTM_MODEL_PATH
)

print("LSTM model loaded successfully.")


# ============================================================
# LOAD BiLSTM
# ============================================================

bilstm_model = load_checkpoint(
    SimpleBiLSTM(),
    BILSTM_MODEL_PATH
)

print("BiLSTM model loaded successfully.")


# ============================================================
# LOAD KAN
# ============================================================

kan_model = load_checkpoint(
    MNISTKAN(),
    KAN_MODEL_PATH
)

print("KAN model loaded successfully.")


# ============================================================
# LOAD ViT
# ============================================================

vit_model = load_checkpoint(
    SimpleViT(),
    VIT_MODEL_PATH
)

print("ViT model loaded successfully.")


# ============================================================
# LOAD GAN ATTACK GENERATOR
# ============================================================

GAN_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "gan_attack_cnn_v2.pth"
)

gan_generator = AttackGenerator(
    epsilon=0.15
).to(device)

gan_generator.load_state_dict(
    torch.load(
        GAN_MODEL_PATH,
        map_location=device,
        weights_only=False
    )
)

gan_generator.eval()

print(
    "GAN attack generator loaded successfully."
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(
        num_output_channels=1
    ),
    transforms.Resize(
        (28, 28)
    ),
    transforms.ToTensor()
])


# ============================================================
# FGSM ATTACK
# ============================================================

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


# ============================================================
# PGD ATTACK
# ============================================================

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

        output = model(
            adversarial_image
        )

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


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message":
        "Adversarial Resilience API is running!"
    }


# ============================================================
# UPLOAD
# ============================================================

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


# ============================================================
# PREDICT + ATTACK
# ============================================================

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("ANN"),
    attack: str = Form("FGSM")
):

    # --------------------------------------------------------
    # Select Model
    # --------------------------------------------------------

    model_name = model_name.upper()
    attack = attack.upper()

    if model_name == "ANN":

        selected_model = ann_model

    elif model_name == "CNN":

        selected_model = cnn_model

    elif model_name == "LSTM":

        selected_model = lstm_model

    elif model_name == "BILSTM":

        selected_model = bilstm_model

    elif model_name == "KAN":

        selected_model = kan_model

    elif model_name == "VIT":

        selected_model = vit_model

    else:

        return {
            "message":
            f"Unsupported model: {model_name}"
        }


    # --------------------------------------------------------
    # Read Image
    # --------------------------------------------------------

    try:

        image_data = await file.read()

        image = Image.open(
            io.BytesIO(image_data)
        ).convert("L")

    except Exception:

        return {
            "message":
            "Invalid image file."
        }


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )


    # --------------------------------------------------------
    # Clean Prediction
    # --------------------------------------------------------

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
        .argmax(
            dim=1
        )
        .item()
    )

    clean_confidence = (
        clean_probabilities[
            0,
            clean_prediction
        ].item()
    )


    # --------------------------------------------------------
    # Attack Label
    # --------------------------------------------------------

    # The actual label of an uploaded
    # image is unknown.
    #
    # Therefore, we use the clean model
    # prediction as the attack target.

    attack_label = torch.tensor(
        [clean_prediction],
        device=device
    )


    # --------------------------------------------------------
    # Attack
    # --------------------------------------------------------

    if attack == "FGSM":

        adversarial_image = fgsm_attack(
            selected_model,
            image_tensor,
            attack_label,
            epsilon=0.25
        )

    elif attack == "PGD":

        adversarial_image = pgd_attack(
            selected_model,
            image_tensor,
            attack_label,
            epsilon=0.25,
            alpha=2 / 255,
            steps=40
        )

    elif attack == "GAN":

        # Current GAN was trained specifically
        # against the CNN model.

        if model_name != "CNN":

            return {
                "message":
                "GAN attack is currently supported only for CNN."
            }

        adversarial_image, _ = (
            generate_adversarial_image(
                gan_generator,
                image_tensor,
                epsilon=0.25
            )
        )

    else:

        return {
            "message":
            f"Unsupported attack: {attack}"
        }


    # --------------------------------------------------------
    # Save Adversarial Image
    # --------------------------------------------------------

    adversarial_filename = (
        f"{uuid.uuid4().hex}.png"
    )

    adversarial_path = os.path.join(
        ADVERSARIAL_DIR,
        adversarial_filename
    )

    adversarial_array = (
        adversarial_image[0]
        .detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    adversarial_array = (
        adversarial_array * 255
    ).clip(
        0,
        255
    ).astype("uint8")

    Image.fromarray(
        adversarial_array
    ).save(
        adversarial_path
    )


    # --------------------------------------------------------
    # Adversarial Prediction
    # --------------------------------------------------------

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
        .argmax(
            dim=1
        )
        .item()
    )

    adversarial_confidence = (
        adversarial_probabilities[
            0,
            adversarial_prediction
        ].item()
    )


    # --------------------------------------------------------
    # Attack Status
    # --------------------------------------------------------

    attack_success = (
        adversarial_prediction
        != clean_prediction
    )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "model": model_name,

        "attack": attack,

        "clean_prediction":
            clean_prediction,

        "clean_confidence":
            round(
                clean_confidence,
                4
            ),

        "adversarial_prediction":
            adversarial_prediction,

        "adversarial_confidence":
            round(
                adversarial_confidence,
                4
            ),

        "attack_success":
            attack_success,

        "adversarial_image_url":
            f"/adversarial_outputs/"
            f"{adversarial_filename}",

        "message":
            "Analysis completed successfully."
    }