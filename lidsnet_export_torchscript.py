import json
import torch
import torch.nn as nn
import os

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


# === Load Model and Mappings ===

with open("lidsnet_feature_map.json", "r", encoding="utf-8") as f:
    feature_map = json.load(f)
label2idx = feature_map["label2idx"]
idx2label = feature_map["idx2label"]
feature_names = feature_map["features"]

input_size = len(feature_names)
hidden_size = 256
num_classes = len(label2idx)

model = LIDSNet(input_dim=input_size, hidden=hidden_size, num_classes=num_classes)
model.load_state_dict(torch.load("lidsnet_model.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
model.eval()

if not os.path.exists("./LIDSNet_torchscript"):
    os.mkdir("./LIDSNet_torchscript")

# Example: adjust dummy input to match real input dimensions (feature vector length)
dummy_input = torch.zeros(1, input_size)  # input_size = len(feature_names)

# Export to traced ScriptModule
traced_script_module = torch.jit.trace(model, dummy_input)
traced_script_module.save("./LIDSNet_torchscript/LIDSNet_intent_detect.pt")  # This file goes to Java side

print("Model exported as TorchScript.")