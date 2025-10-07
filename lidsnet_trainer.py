import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from collections import Counter

# === Configuration ===
JSON_PATH = "fp_symbolic_features.json"
MAX_FEATURES = 2000
BATCH_SIZE = 32
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Dataset ===
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [d["text"] for d in data]
features = [d["features"] for d in data]
labels = [d["label"] for d in data]

# === Encode features ===
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(features)
feature_names = mlb.classes_

# === Encode labels ===
label2idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
idx2label = {idx: label for label, idx in label2idx.items()}
y = [label2idx[label] for label in labels]

# === Dataset & DataLoader ===
class SymbolicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
train_dataset = SymbolicDataset(X_train, y_train)
test_dataset = SymbolicDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# === Simple LIDSNet-style MLP ===
class LIDSNet(nn.Module):
    def __init__(self, input_dim, hidden=256, num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = LIDSNet(input_dim=X.shape[1], num_classes=len(label2idx)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss:.4f}")

# === Evaluation ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y_actual in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        y_true.append(y_actual.item())
        y_pred.append(pred)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=[idx2label[i] for i in range(len(label2idx))]))

# === Save model + mappings (optional) ===
torch.save(model.state_dict(), "lidsnet_model.pt")
with open("lidsnet_feature_map.json", "w") as f:
    json.dump({
        "label2idx": label2idx,
        "idx2label": idx2label,
        "features": feature_names.tolist()
    }, f, indent=2)

