import json
from typing import List, Dict, Set
from preprocessing_layer import PreprocessingLayer
import re

# Load mined FP-Growth patterns
with open("./fp_growth_patterns.json", "r", encoding="utf-8") as f:
    pattern_data = json.load(f)

WH_WORDS = {"what", "where", "when", "why", "how", "who"}

def apply_priority_override(text: str, original_prediction: str) -> str:
    # Lowercased for matching
    lowered = text.lower()

    # Check for WH-word and question mark
    has_wh = any(wh in lowered.split() for wh in WH_WORDS)
    is_question = text.strip().endswith("?")

    # If likely a question, override
    if has_wh and is_question and original_prediction != "ASK_INFORMATION":
        return "ASK_INFORMATION"

    return original_prediction

# Convert patterns to DFA rules
class DFARule:
    def __init__(self, label: str, features: List[str]):
        self.label = label
        self.features = set(features)  # Required feature set

    def matches(self, input_features: Set[str]) -> bool:
        return self.features.issubset(input_features)

# Build DFA rule set
dfa_rules: List[DFARule] = []
for label, patterns in pattern_data.items():
    for pattern in patterns:
        rule = DFARule(label, pattern["itemsets"])
        dfa_rules.append(rule)

# DFA Classifier
def dfa_classify(input_features: List[str]) -> str:
    input_set = set(input_features)
    best_match = None
    max_matched = 0

    for rule in dfa_rules:
        if rule.features.issubset(input_set):
            if len(rule.features) > max_matched:
                best_match = rule.label
                max_matched = len(rule.features)

    return best_match if best_match else "UNKNOWN"


# Initialize the preprocessor
ppl = PreprocessingLayer()

# Sample sentences to test
test_sentences = [
    "Go check that cave behind the hill.",
    "Is it going to rain soon?",
    "This place brings back memories.",
    "Get those diamonds before nightfall!",
    "Where did you last see the zombie?",
    "That was intense!",
    "Would you mind crafting some arrows?",
    "I'm glad you're here.",
    "How do I tame a horse?",
    "Build a tower over there.",
    "Why did the creeper explode?",
    "Let’s rest here for a while.",
    "Could you place a torch here?",
    "This biome is breathtaking.",
    "Can you tell me where spawn is?",
    "I'm kind of nervous about exploring alone.",
    "Craft me a new sword, please.",
    "Who built this underground base?",
    "Let’s enjoy the view for a bit.",
    "Please go explore the nearby cave."
]


# Process and classify using DFA
print("\n🧪 DFA Classification Results (via PreprocessingLayer):")
for sentence in test_sentences:
    features = ppl.process(sentence)
    prediction = dfa_classify(features)  # assuming your DFA function is named this
    final_prediction = apply_priority_override(sentence, prediction)
    print(f"• \"{sentence}\" → Predicted: {final_prediction}")