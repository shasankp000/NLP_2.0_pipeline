import json
from typing import List, Optional


class TuringMachine:
    def __init__(self):
        self.transitions = {}
        self.halting_states = {"q_end_request", "q_end_info", "q_end_general", "q_fallback"}

    def add_transition(self, current_state: str, read_symbol: str, next_state: str):
        self.transitions[(current_state, read_symbol)] = next_state

    def run(self, tape: List[str]) -> str:
        current_state = "q_start"
        head = 0

        while head < len(tape):
            symbol = tape[head]
            key = (current_state, symbol)
            if key in self.transitions:
                next_state = self.transitions[key]
                print(f"{current_state} --[{symbol}]--> {next_state}")  # Debug
                current_state = next_state
                head += 1
            else:
                break

        if current_state in self.halting_states:
            return self.map_state_to_label(current_state)
        else:
            return "UNKNOWN"

    @staticmethod
    def map_state_to_label(state: str) -> str:
        return {
            "q_end_request": "REQUEST_ACTION",
            "q_end_info": "ASK_INFORMATION",
            "q_end_general": "GENERAL_CONVERSATION",
            "q_fallback": "UNKNOWN"
        }.get(state, "UNKNOWN")


# === Load fp_growth_patterns.json ===
with open("fp_growth_patterns.json", "r", encoding="utf-8") as f:
    fp_patterns = json.load(f)

# === Create Turing Machine and build transitions from patterns ===
tm = TuringMachine()

# Define mapping from label to end state
label_to_state = {
    "REQUEST_ACTION": "q_end_request",
    "ASK_INFORMATION": "q_end_info",
    "GENERAL_CONVERSATION": "q_end_general"
}

# Add transitions from mined frequent patterns
for label, patterns in fp_patterns.items():
    for pattern in patterns:
        current_state = "q_start"
        for i, symbol in enumerate(pattern["itemsets"]):
            next_state = f"q_{label.lower()}_{i}"
            tm.add_transition(current_state, symbol, next_state)
            current_state = next_state
        # Final state
        tm.add_transition(current_state, "q_end", label_to_state[label])

# === Test Examples ===
examples = {
    "Could you build a house near the village?": [
        "WH=could", "POS=PRP", "lemma=you", "POS=VB", "lemma=build", "POS=DT", "POS=NN", "lemma=house"
    ],
    "What time is it in the game?": [
        "WH=what", "POS=NN", "POS=VBZ", "POS=IN", "POS=DT", "POS=NN"
    ],
    "The sunset is beautiful today.": [
        "POS=DT", "POS=NN", "lemma=beautiful"
    ]
}

# === Run Classification ===
print("\n🧪 DFA Classification Results (Turing Machine):")
results = {}
for text, tape in examples.items():
    prediction = tm.run(tape + ["q_end"])
    results[text] = prediction
    print(f"• \"{text}\" → Predicted: {prediction}")
