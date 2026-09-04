import os


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms


app = FastAPI(title="Adversarial Resilience API")


# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# ANN Model
# -----------------------------
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


# -----------------------------
# Load trained ANN
# -----------------------------
model = SimpleANN().to(device)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "ann.pth")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Adversarial Resilience API is running!"
    }


# -----------------------------
# Upload endpoint
# -----------------------------
@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    model: str = Form(...),
    attack: str = Form(...)
):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    return {
        "filename": file.filename,
        "width": image.width,
        "height": image.height,
        "model": model,
        "attack": attack,
        "message": "Image, model, and attack received successfully!"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    image_data = await file.read()

    image = Image.open(
        io.BytesIO(image_data)
    ).convert("L")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    predicted_digit = output.argmax(dim=1).item()

    probabilities = torch.softmax(output, dim=1)

    confidence = probabilities[0][predicted_digit].item()

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4)
    }