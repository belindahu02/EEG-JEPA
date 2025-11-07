import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# === Import your own project modules ===
from data_loader import create_session_based_dataloaders
from backbones import SpectrogramResNet  # or LightweightSpectrogramResNet

# --- CONFIG ---
DATA_PATH = "/app/data/grouped_embeddings_full"
USER_IDS = [1, 2, 3]  # make sure these match your dataset class indices
NUM_CLASSES = len(USER_IDS)
INPUT_CHANNELS = 1
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 300  # should be enough to overfit 196 samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# --- CREATE DATALOADERS ---
train_loader, val_loader, test_loader, _ = create_session_based_dataloaders(
    data_path=DATA_PATH,
    user_ids=USER_IDS,
    normalization="none",
    batch_size=BATCH_SIZE,
    augment_train=False,
    cache_size=50
)

# Count total samples
num_train_samples = sum(inputs.size(0) for inputs, _ in train_loader)
print(f"Train loader contains {num_train_samples} samples")

# --- MODEL / LOSS / OPTIMIZER ---
model = SpectrogramResNet(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- TRAIN LOOP OVER ENTIRE TRAIN SET ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = epoch_loss / total
    acc = correct / total

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

# --- FINAL CHECK ON ENTIRE TRAINING SET ---
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

print("\n✅ Final results on entire training set:")
print("Preds :", all_preds)
print("Target:", all_targets)

# Optional: final accuracy
final_acc = (all_preds == all_targets).mean()
print(f"Final training accuracy: {final_acc:.4f}")
