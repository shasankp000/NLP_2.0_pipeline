import json
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from collections import defaultdict

# Load symbolic feature data
with open("./fp_symbolic_features.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Separate features per class
class_transactions = defaultdict(list)
for entry in dataset:
    label = entry["label"]
    features = entry["features"]
    class_transactions[label].append(features)

# Config
MIN_SUPPORT = 0.3  # Adjust based on dataset size and needs
TOP_N = 20         # Number of top frequent patterns to display/export

# Store mined patterns
all_frequent_patterns = {}

for label, transactions in class_transactions.items():
    print(f"\n🧩 Mining patterns for class: {label}")

    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Mine frequent itemsets
    patterns = fpgrowth(df, min_support=MIN_SUPPORT, use_colnames=True)
    sorted_patterns = patterns.sort_values("support", ascending=False).head(TOP_N)

    # Convert frozenset to sorted list for JSON
    converted_patterns = []
    for row in sorted_patterns.to_dict(orient="records"):
        row["itemsets"] = sorted(list(row["itemsets"]))
        converted_patterns.append(row)

    all_frequent_patterns[label] = converted_patterns

    # Show preview
    print(pd.DataFrame(converted_patterns))

# Save to JSON
with open("fp_growth_patterns.json", "w", encoding="utf-8") as f:
    json.dump(all_frequent_patterns, f, indent=2)

print("\n✅ Frequent patterns saved to fp_growth_patterns.json")
