import spacy
import random
import json
from itertools import combinations
from collections import Counter
from typing import List, Tuple, Dict
from tqdm import tqdm
import language_tool_python
from preprocessing_layer import PreprocessingLayer

MAX_PER_CLASS = 200

# Load spaCy and grammar tool
nlp = spacy.load("en_core_web_sm")
grammar_tool = language_tool_python.LanguageTool('en-US')

# === Use single instance of preprocessor
preprocessor = PreprocessingLayer()

# ===== Dataset =====
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

# === Grammar-aware Synthetic Generator ===
def extract_chunks(sentence: str):
    doc = nlp(sentence)
    wh = [tok.text for tok in doc if tok.tag_ in ["WDT", "WP", "WRB"]]
    verbs = [tok.lemma_ for tok in doc if tok.pos_ == "VERB"]
    nouns = [tok.text for tok in doc if tok.pos_ in ["NOUN", "PROPN"]]
    return wh, verbs, nouns

def combine_sentences(s1: str, s2: str):
    wh1, v1, n1 = extract_chunks(s1)
    wh2, v2, n2 = extract_chunks(s2)
    wh = random.choice(wh1 or wh2 or ["What"])
    verb = random.choice(v1 or v2 or ["is"])
    noun = random.choice(n1 + n2 or ["thing"])
    return f"{wh} {verb} the {noun}?"

def correct_grammar(sentence: str):
    matches = grammar_tool.check(sentence)
    return language_tool_python.utils.correct(sentence, matches)

# === Per-Class Augmentation ===
def augment_sentences(class_name: str, samples: List[str]) -> List[str]:
    phrases = emotion_phrases.get(class_name, [])
    augmented = []
    for sent in samples:
        for phrase in phrases:
            if phrase.endswith(",") or phrase.endswith(":"):
                augmented.append(f"{phrase} {sent}")
            else:
                augmented.append(f"{phrase} {sent}")
    return list(set(augmented))

# === Synthetic Combination Expansion ===
def generate_synthetic_sentences(label: str, base_samples: List[str], count: int) -> List[Tuple[str, str]]:
    pairs = list(combinations(base_samples, 2))
    random.shuffle(pairs)
    combined = []
    for s1, s2 in pairs[:count]:
        imp = combine_sentences(s1, s2)
        corr = correct_grammar(imp)
        combined.extend([(imp, label), (corr, label)])
    return combined


# === Consistent FP-Growth feature extraction using PreprocessingLayer
def extract_fp_features_via_preprocessor(text: str) -> List[str]:
    data = preprocessor.process(text)
    features = []

    for lemma_pair in data["lemmas"]:
        word, lemma = lemma_pair
        features.append(f"lemma={lemma}")

    for pos_pair in data["pos_tags"]:
        word, tag = pos_pair
        features.append(f"POS={tag}")
        if tag in ["WDT", "WP", "WRB", "WP$", "MD"]:
            features.append(f"WH={word.lower()}")

    for ent in data["entities"]:
        features.append(f"NER={ent['label']}")

    return sorted(set(features))

# === Build symbolic dataset
symbolic_dataset = []

for label, base_samples in tqdm(examples.items(), desc="Processing Classes"):
    # Original + Augmented examples
    originals = [(s, label) for s in base_samples]
    augmented = [(s, label) for s in augment_sentences(label, base_samples)]
    combined = originals + augmented

    # Calculate how many synthetic samples are needed to hit MAX_PER_CLASS
    current_count = len(combined)
    needed = max(0, MAX_PER_CLASS - current_count)
    synthetic_pairs = (needed + 1) // 2  # Each synthetic pair gives 2 samples (imperfect + corrected)

    # Generate enough synthetic data to reach the cap
    synthetic = generate_synthetic_sentences(label, [s for s, _ in combined], synthetic_pairs)
    all_examples = (combined + synthetic)[:MAX_PER_CLASS]

    for text, tag in all_examples:
        feats = extract_fp_features_via_preprocessor(text)
        symbolic_dataset.append({
            "text": text,
            "label": tag,
            "features": feats
        })


# === Export
with open("fp_symbolic_features.json", "w", encoding="utf-8") as f:
    json.dump(symbolic_dataset, f, indent=2)

print(f"✅ Exported {len(symbolic_dataset)} symbolic feature entries.")
print("📊 Class breakdown:", Counter(d["label"] for d in symbolic_dataset))
