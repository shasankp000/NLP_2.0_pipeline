import json
import numpy as np
from preprocessing_layer import PreprocessingLayer

# === Load JSON Files ===
with open("cart_tree.json", "r") as f:
    tree = json.load(f)

with open("cart_vectorizer_vocab.json", "r") as f:
    feature_index = json.load(f)

with open("cart_class_labels.json", "r") as f:
    class_labels = json.load(f)


# === Preprocessing ===
preprocessor = PreprocessingLayer()

def encode_features(features):
    vec = np.zeros(len(feature_index), dtype=np.float32)
    for feat in features:
        if feat in feature_index:
            vec[feature_index[feat]] = 1.0
    return vec


# === Recursive Prediction Function ===
def predict_from_tree(tree_node, feature_vector):
    if tree_node["type"] == "leaf":
        return class_labels[tree_node["class"]], tree_node["confidence"]
    else:
        feat_name = tree_node["feature"]
        feat_idx = feature_index.get(feat_name, None)
        if feat_idx is None:
            # Feature missing — go default to left
            return predict_from_tree(tree_node["left"], feature_vector)
        if feature_vector[feat_idx] <= tree_node["threshold"]:
            return predict_from_tree(tree_node["left"], feature_vector)
        else:
            return predict_from_tree(tree_node["right"], feature_vector)


# === Test Inputs ===
test_samples = [
    # REQUEST_ACTION
    "Please build a tower beside the river.",
    "Dig straight down until you reach bedrock.",
    "Can you light up the cave system?",
    "Quickly place torches around the base!",
    "Mine all the coal you find.",
    "Craft a sword and defend me now!",
    "Put a chest down near the entrance.",
    "Go back to the village immediately.",

    # ASK_INFORMATION
    "Where did you go last night?",
    "Can you tell me your health status?",
    "Why are there so many mobs nearby?",
    "How much iron do we have left?",
    "What time is it in-game?",
    "Who built this beautiful structure?",
    "Do you know if we've passed the village?",
    "Could you check the crafting recipe for an anvil?",

    # GENERAL_CONVERSATION
    "This forest looks absolutely stunning.",
    "I love building with you.",
    "That was a fun mining trip!",
    "Honestly, this server feels like home.",
    "Sometimes I just want to sit and watch the sunrise.",
    "Our adventure has been amazing so far.",
    "That creeper gave me a heart attack.",
    "To be honest, I enjoy talking to you more than playing."
]


# === Predict ===
for text in test_samples:
    features = preprocessor.process(text)
    vector = encode_features(features)
    prediction, confidence = predict_from_tree(tree, vector)
    print(f"🗣 \"{text}\"\n➡️  Predicted class: {prediction} (confidence: {confidence:.2f})\n")
