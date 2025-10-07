from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import numpy as np
from datasets import load_dataset
from preprocessing_layer import PreprocessingLayer
from sklearn.preprocessing import MultiLabelBinarizer

# Load test.json
# Load and flatten the dataset
with open("test.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Convert dict into list of (text, label) pairs
final_dataset = [(text, label) for label, samples in raw_data.items() for text in samples]


# === CART Classifier ===
print("Training CART classifier....")

print("🔍 Using symbolic features (lemma + POS) for CART training...")

# Initialize symbolic processor
symbolic_processor = PreprocessingLayer()

# Extract symbolic features
# final_dataset is a list of (text, label) tuples
X_symbolic = [symbolic_processor.process(text) for text, _ in final_dataset]
y_cart = [label for _, label in final_dataset]

# One-hot encode symbolic features
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X_symbolic)

# Train CART
clf = DecisionTreeClassifier(max_depth=6)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_cart, test_size=0.2, stratify=y_cart)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nCART Evaluation:\n", classification_report(y_test, y_pred))

# Print rules
print("\nCART Rules:\n", export_text(clf, feature_names=mlb.classes_))

for i, sym in enumerate(X_symbolic[:5]):
    print(f"[Sample {i}] Features: {sym}")



def export_cart_tree(clf, feature_names):
    tree = clf.tree_

    def recurse(node):
        if tree.feature[node] == -2:  # Leaf node
            class_counts = tree.value[node][0].tolist()
            total = sum(class_counts)
            predicted_class = int(np.argmax(class_counts))
            confidence = max(class_counts) / total if total > 0 else 0.0

            return {
                "type": "leaf",
                "class": predicted_class,
                "class_counts": class_counts,
                "confidence": round(confidence, 4)
            }
        else:
            return {
                "type": "split",
                "feature": feature_names[tree.feature[node]],
                "threshold": float(tree.threshold[node]),
                "left": recurse(tree.children_left[node]),
                "right": recurse(tree.children_right[node])
            }

    return recurse(0)



print("Exporting CART tree to json files...")


# Save outputs
print("Exporting symbolic CART tree to JSON...")

with open("cart_tree.json", "w") as f:
    json.dump(export_cart_tree(clf, mlb.classes_), f, indent=2)

with open("cart_vectorizer_vocab.json", "w") as f:
    json.dump({feat: idx for idx, feat in enumerate(mlb.classes_)}, f, indent=2)

with open("cart_class_labels.json", "w") as f:
    json.dump(clf.classes_.tolist(), f, indent=2)

print("✅ Export complete using symbolic features.")