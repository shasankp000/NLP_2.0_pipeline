import random
from collections import defaultdict, Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import spacy
import language_tool_python
from itertools import combinations
from typing import List, Tuple
import json
import numpy as np
from datasets import load_dataset
from preprocessing_layer import PreprocessingLayer
from sklearn.preprocessing import MultiLabelBinarizer

# === NLP & Grammar Tools ===
print("Loading grammar tool....")

nlp = spacy.load("en_core_web_sm")
grammar_tool = language_tool_python.LanguageTool('en-US')

print("Grammar tool loaded.")



# === Dataset ===
examples = {
    "REQUEST_ACTION": [
        "Could you build a house near the village?",
        "Go to the cave and check for ores.",
        "Please craft a diamond sword.",
        "Walk to the nearest village.",
        "Run towards the mountain.",
        "Mine some iron ore for me.",
        "Can you dig a tunnel here?",
        "Will you attack any zombies nearby?",
        "Navigate through the forest.",
        "Craft a set of armor.",
        "Could you farm some wheat?",
        "Plant some trees around here.",
        "Go and explore the desert biome.",
        "Hunt for some food, please.",
        "Defend the base if mobs attack.",
        "Build a bridge across the river.",
        "Dig down to find diamonds.",
        "Set up a small shelter here.",
        "Collect wood and stone.",
        "Travel to the coordinates I gave you.",
        "dig straight down",
        "make a quick base",
        "build farm here",
        "go there fast",
        "light up cave",
        "mine all iron u see",
        "grab wood from forest",
        "dig that block",
        "get ready mobs coming",
        "make safehouse now",
        "put torch here",
        "go find lava",
        "make sword quick",
        "place bed nearby",
        "get that coal",
        "craft shield",
        "plant saplings around",
        "go to cords 100 64 -20",
        "build wall fast",
        "kill skeleton behind me",
        "mine here pls",
        "dig tunnel to village",
        "put chest down",
        "find diamond soon",
        "farm this area",
        "get ready defend base",
        "create furnace now",
        "dig down 5 blocks",
        "place stairs up",
        "smelt iron asap"
    ],
    "ASK_INFORMATION": [
        "What time is it in the game?",
        "Where is the nearest village?",
        "Who built this structure?",
        "Why are there so many mobs tonight?",
        "How much health do you have?",
        "Did you find any diamonds yet?",
        "When will it stop raining?",
        "Can you tell me the recipe for a golden apple?",
        "Where did you get that armor?",
        "How far are we from spawn?",
        "Do you know where my house is?",
        "What do you see around you?",
        "Could you tell me your coordinates?",
        "Is it safe to go outside now?",
        "Who is online right now?",
        "How long have you been exploring?",
        "Where did you last see the creeper?",
        "Did you check the chest?",
        "What is the best way to find iron?",
        "Can you explain how redstone works?",
    ],
    "GENERAL_CONVERSATION": [
        "I love building castles.",
        "This forest looks amazing.",
        "The sunset is beautiful today.",
        "I found a cool cave yesterday.",
        "Exploring the ocean is fun.",
        "It’s peaceful around here.",
        "I really like this server.",
        "My friend joined the game earlier.",
        "The weather is so clear right now.",
        "I’m going to build a big farm.",
        "Let’s relax and watch the sunset.",
        "Sometimes it’s nice just to walk around.",
        "I enjoy talking with you.",
        "I had fun mining today.",
        "The base looks great now.",
        "This game is so relaxing.",
        "Building with you is fun.",
        "Our adventure has been awesome.",
        "I love how the village turned out.",
        "That was a good fight back there."
    ]
}

# ===== Class-based Emotional Augmentation =====
emotion_phrases = {
    "REQUEST_ACTION": ["Please", "Could you", "Would you kindly", "Quickly", "If you can"],
    "ASK_INFORMATION": ["I'm wondering,", "Do you know", "Can I ask", "I'm curious,", "Tell me"],
    "GENERAL_CONVERSATION": ["Honestly,", "To be honest,", "I feel like", "Sometimes", "I really think"]
}


# === Synthetic Sentence Generation + Grammar Correction ===
# === Synthetic Sentence Generation + Grammar Correction ===
def extract_chunks(sentence):
    doc = nlp(sentence)
    wh = [tok.text for tok in doc if tok.tag_ in ["WDT", "WP", "WRB"]]
    verbs = [tok.lemma_ for tok in doc if tok.pos_ == "VERB"]
    nouns = [tok.text for tok in doc if tok.pos_ in ["NOUN", "PROPN"]]
    return wh, verbs, nouns

def combine_sentences(s1, s2):
    wh1, v1, n1 = extract_chunks(s1)
    wh2, v2, n2 = extract_chunks(s2)
    wh = random.choice(wh1 or wh2 or ["What"])
    verb = random.choice(v1 or v2 or ["is"])
    noun = random.choice(n1 + n2 or ["thing"])
    return f"{wh} {verb} the {noun}?"

def correct_grammar(sentence):
    matches = grammar_tool.check(sentence)
    return language_tool_python.utils.correct(sentence, matches)

# === Per-Class Augmentation ===
def augment_sentences(class_name: str, samples: List[str]) -> List[str]:
    phrases = emotion_phrases.get(class_name, [])
    augmented = []
    for sent in samples:
        for phrase in phrases:
            augmented.append(f"{phrase} {sent}")
    return list(set(augmented))

def generate_balanced_synthetic(label, base_samples, count):
    if len(base_samples) < 2:
        return []  # Can't form combinations

    pairs = list(combinations(base_samples, 2))
    if not pairs:
        return []

    random.shuffle(pairs)
    selected_pairs = pairs[:min(count, len(pairs))]

    imperfect, corrected = [], []
    for s1, s2 in selected_pairs:
        imp = combine_sentences(s1, s2)
        corr = correct_grammar(imp)
        imperfect.append((imp, label))
        corrected.append((corr, label))

    return imperfect + corrected




# === Map CLINC150 intents to your 3 classes ===
intent_mapping = {
    "REQUEST_ACTION": [
        "set_alarm", "transfer_money", "order", "pay_bill", "book_flight", "book_hotel", "get_weather",
        "cancel", "cook_recipe", "turn_on", "turn_off"
    ],
    "ASK_INFORMATION": [
        "what_is_your_name", "weather", "track_package", "flight_status", "restaurant_reservation",
        "calendar", "time", "current_location", "translate", "meaning_of_life"
    ],
    "GENERAL_CONVERSATION": [
        "greeting", "thank_you", "goodbye", "tell_joke", "small_talk", "fun_fact"
    ]
}

# === Load CLINC dataset ===
clinc_data = load_dataset("clinc_oos", "small")
new_examples = defaultdict(list)

# === Filter and assign samples ===
for split in ["train", "validation"]:
    for item in clinc_data[split]:
        intent_label = clinc_data["train"].features["intent"].int2str(item["intent"])
        text = item["text"].strip()

        for class_name, intents in intent_mapping.items():
            if intent_label in intents:
                new_examples[class_name].append(text)
                break

# === Reduce & deduplicate ===
for class_name in new_examples:
    new_examples[class_name] = list(set(new_examples[class_name]))[:100]  # limit to 100 max

# === Merge into your existing examples ===
for class_name, extra_samples in new_examples.items():
    examples[class_name].extend(extra_samples)

# === Expand Dataset ===
print("Expanding dataset with synthetic data....")

final_dataset = []
for label, base in tqdm(examples.items(), desc="Processing labels"):
    original = [(text, label) for text in base]
    augmented = augment_sentences(label, [text for text, _ in original])
    combined = original + [(text, label) for text in augmented]

    print(f"🧪 Class '{label}' - base: {len(base)}, augmented: {len(augmented)}, combined: {len(combined)}")


    if len(combined) < 2:
        print(f"⚠️ Skipping synthetic generation for class '{label}' due to insufficient samples ({len(combined)})")
        synthetic = []
    else:
        synthetic = generate_balanced_synthetic(label, [s for s, _ in combined], 20)

    final_dataset.extend(combined + synthetic)


print("Dataset expansion complete.")


print("Final dataset size:", len(final_dataset))
print("Label distribution:", Counter(label for _, label in final_dataset))

# Convert to a list of dicts
json_data = [{"text": text, "label": label} for text, label in final_dataset]

# Save to file
with open("final_dataset_raw.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print("✅ Saved final_dataset to final_dataset_raw.json")



# # === CART Classifier ===
# print("Training CART classifier....")

# print("🔍 Using symbolic features (lemma + POS) for CART training...")

# # Initialize symbolic processor
# symbolic_processor = PreprocessingLayer()

# # Extract symbolic features
# X_symbolic = [symbolic_processor.process(text) for text, _ in final_dataset]
# y_cart = [label for _, label in final_dataset]

# # One-hot encode symbolic features
# mlb = MultiLabelBinarizer()
# X_encoded = mlb.fit_transform(X_symbolic)

# # Train CART
# clf = DecisionTreeClassifier(max_depth=6)
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_cart, test_size=0.2, stratify=y_cart)
# clf.fit(X_train, y_train)

# # Evaluate
# y_pred = clf.predict(X_test)
# print("\nCART Evaluation:\n", classification_report(y_test, y_pred))

# # Print rules
# print("\nCART Rules:\n", export_text(clf, feature_names=mlb.classes_))

# for i, sym in enumerate(X_symbolic[:5]):
#     print(f"[Sample {i}] Features: {sym}")




# def export_cart_tree(clf, feature_names):
#     tree = clf.tree_

#     def recurse(node):
#         if tree.feature[node] == -2:  # Leaf node
#             class_counts = tree.value[node][0].tolist()
#             total = sum(class_counts)
#             predicted_class = int(np.argmax(class_counts))
#             confidence = max(class_counts) / total if total > 0 else 0.0

#             return {
#                 "type": "leaf",
#                 "class": predicted_class,
#                 "class_counts": class_counts,
#                 "confidence": round(confidence, 4)
#             }
#         else:
#             return {
#                 "type": "split",
#                 "feature": feature_names[tree.feature[node]],
#                 "threshold": float(tree.threshold[node]),
#                 "left": recurse(tree.children_left[node]),
#                 "right": recurse(tree.children_right[node])
#             }

#     return recurse(0)



# print("Exporting CART tree to json files...")


# # Save outputs
# print("Exporting symbolic CART tree to JSON...")

# with open("cart_tree.json", "w") as f:
#     json.dump(export_cart_tree(clf, mlb.classes_), f, indent=2)

# with open("cart_vectorizer_vocab.json", "w") as f:
#     json.dump({feat: idx for idx, feat in enumerate(mlb.classes_)}, f, indent=2)

# with open("cart_class_labels.json", "w") as f:
#     json.dump(clf.classes_.tolist(), f, indent=2)

# print("✅ Export complete using symbolic features.")