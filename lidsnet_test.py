import torch
import torch.nn as nn
import json
from preprocessing_layer import PreprocessingLayer # Your symbolic feature extractor
from typing import List

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

# === Preprocessing setup ===

processor = PreprocessingLayer()

def text_to_vector(text):
    features = processor.process(text)  # or use .extract_fp_features if that's your function name
    vec = torch.zeros(len(feature_names))
    # Use feature_names for order; features from extraction may not be present!
    for i, feat in enumerate(feature_names):
        if feat in features:
            vec[i] = 1.0
    return vec.unsqueeze(0)  # Add batch dimension

def classify(texts):
    print("🧪 LIDSNet Inference Results:")
    for text in texts:
        input_vec = text_to_vector(text)
        with torch.no_grad():
            logits = model(input_vec)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        print(f'• "{text}" → Predicted: {idx2label[str(pred)]} (Confidence: {confidence:.2f})')

# === Sample tests ===

test_inputs = [
    "Could you build a house near the village?",
    "What time is it in the game?",
    "The sunset is beautiful today.",
    "Mine 3 blocks down please.",
    "Where are you now?",
    "That was a fun adventure."
]

stress_test_inputs = [
    # Ambiguous and Blended Intents
    "Can you help me build something or just talk for a bit?",
    "What's over there?",
    "Tell me a joke and then mine three blocks.",
    "Let's see... should I dig or do you think it’s nighttime?",
    "What are you doing right now?",

    # Unusual Syntax, Typos, and Slang
    "yo buildhouse plz",
    "izzit raining in minecraft??",
    "Mien the trez beside me, kthx",
    "gimme torchs, bud",
    "wut you up to?",

    # Conversational Small Talk & Idle Chatter
    "You're really helpful, thanks!",
    "This place is so peaceful.",
    "How do you feel today?",
    "Haha, nice one!",
    "That was epic, wasn’t it?",

    # Indirect or Polite Requests
    "Would you mind grabbing some wood for me?",
    "I was wondering if you could show me where the village is.",
    "Could you maybe light up the cave?",
    "I'd appreciate some info on my coordinates.",

    # Multi-step Commands / Follow-ups
    "Dig a tunnel there and then tell me how long it is.",
    "After building the wall, check the time.",
    "Can you lead the way and explain what you're doing?",
    "Mine and if you find iron, say so.",

    # Vague, Out-of-Scope, or No Real Intent
    "Purple elephants fly at dawn.",
    "Banana?",
    "I like turtles.",
    "Can you do the thing?",
    "ehhhh…",

    # Clarifying and Meta-Commands
    "Repeat the last thing you did.",
    "Ignore my last command.",
    "What did I ask you before?",
    "Help. I messed up.",

    # Blended In-Game/Out-of-Game Context
    "What version is this server running?",
    "Do you remember what biome we started in?",
    "Type /seed for me, please."
]


classify(test_inputs)
classify(stress_test_inputs)
