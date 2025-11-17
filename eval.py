import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_preds(csv_path):
    gt, pred = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt.append(row["Ground Truth"])
            pred.append(row["Prediction"])
    return gt, pred

def main():
    csv_path = "outputs/predictions.csv"
    gt, pred = load_preds(csv_path)
    labels = sorted(set(gt) | set(pred))
    cm = confusion_matrix(gt, pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix from CSV Predictions")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_from_csv.png", dpi=300)
    plt.show()
    acc = np.mean(np.array(gt) == np.array(pred))
    print(f"Accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()
