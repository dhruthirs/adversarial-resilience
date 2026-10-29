# Adversarial Resilience

An MNIST research project for generating adversarial examples and measuring how
different restoration methods recover classifier predictions. The project has a
FastAPI backend, a React/Vite frontend, six MNIST classifiers, adversarial attack
generators, B-spline/wavelet restoration, WGAN-GP restoration, and comparison
scripts.

## What The Project Does

The system follows this overall flow:

```text
MNIST image
	|
	v
Classifier prediction
	|
	v
FGSM, PGD, or GAN attack
	|
	v
Adversarial image
	|
	+--> No restoration ------------> Classifier
	+--> B-spline restoration ------> Classifier
	+--> WGAN restoration ----------> Classifier
	+--> B-spline + WGAN cascade ---> Classifier
```

Images are converted to one grayscale `28 x 28` tensor with values in `[0, 1]`.
The application supports these classifiers:

- `ANN`: fully connected network with `784 -> 512 -> 256 -> 10` layers.
- `CNN`: two convolution/max-pooling blocks followed by a classifier.
- `LSTM`: two-layer sequence model over MNIST rows.
- `BiLSTM`: bidirectional version of the LSTM model.
- `KAN`: Kolmogorov-Arnold Network with hidden sizes `[784, 64, 10]`.
- `ViT`: vision transformer with `7 x 7` patches, depth `4`, and `4` heads.

## Repository Layout

```text
backend/                 FastAPI API, attacks, models, training, evaluation
models/                  Additional model definitions
efficient_kan/           Local KAN implementation
evaluation/              Comparison benchmark and CSV/chart generation
frontend/                React/Vite user interface
weights/                 Trained model checkpoints
data/                    Downloaded MNIST data, created when needed
adversarial_outputs/     Generated adversarial and restored PNG files
notebooks/               Exploratory notebooks
python-dependencies.dot  Python import graph, not a pip requirements file
frontend-dependencies.dot React import graph
```

## Environment Setup

The commands below are run from the repository root:

```powershell
cd D:\mini2final\adversarial-resilience
python -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the CUDA build for an NVIDIA GPU such as the RTX 4050:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining Python packages:

```powershell
python -m pip install fastapi uvicorn python-multipart pillow vit-pytorch matplotlib torchattacks numpy
```

Verify the GPU environment:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The current verified environment uses `torch 2.11.0+cu128` and an NVIDIA RTX
4050 Laptop GPU. Training scripts require CUDA and raise an error when CUDA is
not available. The API and evaluation scripts can fall back to CPU, but will be
much slower.

The frontend uses Node.js and Vite:

```powershell
cd frontend
npm install
```

`python-dependencies.dot` and `frontend-dependencies.dot` describe imports for
Graphviz. They are documentation files and must not be passed to `pip install`.

## Trained Checkpoints

The backend expects these files under `weights/`:

| File | Purpose |
| --- | --- |
| `ann.pth` | ANN classifier |
| `cnn.pth` | CNN classifier |
| `lstm.pth` | LSTM classifier |
| `bilstm.pth` | BiLSTM classifier |
| `kan.pth` | KAN classifier |
| `vit.pth` | ViT classifier |
| `gan_attack_all.pth` | GAN attack trained across the classifier family |
| `gan_attack_cnn_v2.pth` | CNN-specific GAN attack fallback |
| `restorer.pth` | Trained B-spline/wavelet restorer |
| `wgan_restorer.pth` | WGAN-GP restoration generator |

The API requires `restorer.pth` for B-spline or combined restoration and
`wgan_restorer.pth` for WGAN or combined restoration. It does not silently use
an untrained restoration model.

## Attack Pipelines

### FGSM

FGSM computes one input-gradient step:

```text
image + epsilon * sign(gradient of classifier loss)
```

The API uses `epsilon=0.25`. The comparison benchmark uses `epsilon=0.30`.

### PGD

PGD repeats signed gradient steps and projects the result into an epsilon ball:

```text
start with clean image
repeat steps times:
	image += alpha * sign(input gradient)
	project image into clean_image +/- epsilon
	clamp pixels to [0, 1]
```

The API uses `epsilon=0.25`, `alpha=2/255`, and `40` steps. The comparison
benchmark uses `epsilon=0.30`, `alpha=2/255`, and `40` steps with a random start.

### GAN Attack

`AttackGenerator` predicts a bounded perturbation using a convolutional encoder
and decoder. Its output uses `tanh` and is limited to the configured epsilon.
The multi-classifier attack is trained by `backend/train_gan_all.py` and saved as
`weights/gan_attack_all.pth`.

The API loads the generator with training epsilon `0.15` and rescales its
perturbation to inference epsilon `0.25`. The comparison benchmark trains and
evaluates the generator at epsilon `0.30`.

## Restoration Pipelines

### B-spline/Wavelet Restoration

Implemented by `WaveletSplineRestorer` in `backend/restoration.py`:

```text
Adversarial image
	|
	+--> Cubic B-spline projection
	+--> Mexican-hat wavelet response
	+--> Morlet wavelet response
	+--> Learned convolutional refinement
	|
	v
Restored image
```

The default projection strength is `0.85` and wavelet strength is `0.10`.
The trainable refinement network uses convolution channels `32 -> 32 -> 1`.

`backend/train_restorer.py` trains this model for `10` epochs with batch size
`256`, AdamW learning rate `1e-3`, and attack epsilon `0.30`. Training rotates
through FGSM, PGD, and GAN-generated examples. Its loss is:

```text
MSE(restored, clean) + 0.1 * average_classifier_cross_entropy
```

### WGAN-GP Restoration

`WGANRestorer` wraps a `WaveletSplineRestorer` as its generator. A separate
`RestorationCritic` learns to distinguish restored images from clean images.

```text
Adversarial image --> WGANRestorer generator --> restored image
									  |
									  v
							  RestorationCritic
```

`backend/train_wgan_restorer.py` uses `10` epochs, batch size `256`, epsilon
`0.30`, and AdamW learning rate `1e-4` for both generator and critic. Most
batches use 40-step PGD; every third batch uses the multi-classifier GAN attack.

The generator loss is:

```text
-critic(restored)
+ 100.0 * MSE(restored, clean)
+ 0.1 * average_classifier_cross_entropy
```

The critic uses the Wasserstein loss plus gradient penalty weight `10.0`.

### B-spline + WGAN Cascade

The combined method is a per-sample cascade, not an unconditional second
filter:

```text
Adversarial image
	|
	+--> B-spline output ------------------+
	|                                       |
	+--> WGAN(B-spline output) ------------+
											v
								  Classify both candidates
											|
											v
								  Select final candidate
```

The selector in `backend/restoration.py` follows these rules:

1. Prefer a candidate whose prediction matches the clean-image prediction.
2. If both candidates match, choose the one with higher softmax confidence.
3. If neither matches, choose the higher-confidence candidate.

The API applies this selector in `backend/main.py`. The benchmark applies the
same selector in `evaluation/comparison_chart.py`, so the displayed application
and reported CSV use the same combined behavior.

## Training Order

Run the training stages in this order when rebuilding checkpoints:

```powershell
# Train the six classifiers first.
python backend\train_ann.py
python backend\train_cnn.py
python backend\train_lstm.py
python backend\train_bilstm.py
python backend\train_kan.py
python backend\train_vit.py

# Train the multi-classifier GAN attack.
python backend\train_gan_all.py

# Train B-spline/wavelet restoration.
python backend\train_restorer.py

# Train WGAN-GP restoration.
python backend\train_wgan_restorer.py
```

All restoration and GAN training commands should use the GPU interpreter:

```powershell
.\.venv-gpu\Scripts\python.exe backend\train_restorer.py
.\.venv-gpu\Scripts\python.exe backend\train_wgan_restorer.py
```

Training can take a long time and overwrites the corresponding checkpoint.

## Evaluation

Regenerate the comparison CSV and chart with:

```powershell
.\.venv-gpu\Scripts\python.exe evaluation\comparison_chart.py
```

This evaluates all six classifiers on the MNIST test set using batch size `256`
and benchmark epsilon `0.30`. It writes:

- `evaluation/restoration_comparison.csv`
- `evaluation/restoration_comparison.png`

The CSV columns mean:

- `clean_accuracy`: accuracy on clean test images.
- `*_asr`: attack success rate among initially correct samples.
- `*_accuracy`: accuracy after the corresponding attack or restoration.
- `*_restoration_success`: percentage of successful PGD attacks recovered by the restoration method.
- `*_drop`: clean accuracy minus attacked accuracy.

Restoration success is measured on samples that were originally classified
correctly and then successfully fooled by PGD. The benchmark has true MNIST
labels, so these metrics are ground-truth evaluation metrics.

`backend/evaluate_restorer.py` performs a separate adaptive evaluation where
the attack differentiates through the B-spline restorer. This is a stronger
white-box test and should not be confused with the standard comparison chart.

## Running The Application

Start the backend from the repository root:

```powershell
.\.venv-gpu\Scripts\python.exe -m uvicorn backend.main:app --reload
```

The API listens at `http://127.0.0.1:8000`.

Start the frontend in a second terminal:

```powershell
cd frontend
npm run dev
```

Vite normally serves the UI at `http://localhost:5173`.

### API Endpoints

`GET /` confirms that the API is running.

`POST /upload` accepts an image plus `model` and `attack` form fields and returns
basic upload information.

`POST /predict` accepts these multipart form fields:

| Field | Values | Default |
| --- | --- | --- |
| `file` | Image file | Required |
| `model_name` | `ANN`, `CNN`, `LSTM`, `BILSTM`, `KAN`, `VIT` | `ANN` |
| `attack` | `FGSM`, `PGD`, `GAN` | `FGSM` |
| `restore` | Boolean | `true` |
| `restoration_method` | `BSPLINE`, `WGAN`, `BSPLINE_WGAN` | `BSPLINE` |

The response includes clean, adversarial, and restored predictions/confidences,
attack status, restoration method, mean absolute pixel change, and URLs for the
generated adversarial/restored images.

Uploaded images have no ground-truth label, so the API does not claim that a
restoration is correct against the true label. The cascade uses the clean model
prediction as its selection reference; use the benchmark scripts for
ground-truth accuracy and recovery statistics.

## Dependency Graphs

Render the checked-in Graphviz files with:

```powershell
dot -Tsvg python-dependencies.dot -o python-dependencies.svg
dot -Tsvg frontend-dependencies.dot -o frontend-dependencies.svg
```

The graphs are import documentation only. The detailed graph workflow is in
[graphify.md](graphify.md).