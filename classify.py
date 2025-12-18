import torch
import numpy as np
import cv2
from PIL import Image
import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def _bgr_to_pil(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

@torch.no_grad()
def classify_clip_emb(img_emb, text_stack, labels, agg="mean", tau_other=0.27):
    """
    img_emb: [K, D]   text_stack: [C, D]
    returns (label, score, sims[tensor[C]])
    """
    if img_emb.ndim == 1:
        v = img_emb.unsqueeze(0)
    else:
        if agg == "mean":
            v = img_emb.mean(dim=0, keepdim=True)
        elif agg == "median":
            v = img_emb.median(dim=0).values.unsqueeze(0)
        elif agg == "max":
            v = img_emb.max(dim=0).values.unsqueeze(0)
        else:
            raise ValueError("agg must be mean|median|max")

    sims = (v @ text_stack.T).squeeze(0)  # cosine sims [C]
    best_idx = int(torch.argmax(sims).item())
    best_label = labels[best_idx]
    best_score = float(sims[best_idx].item())
    if best_score < tau_other:
        return "Other", best_score, sims
    return best_label, best_score, sims

def sample_frames(cap, sample_k=16):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # webcam case or unknown length: just read first K frames
        stride = 1
    else:
        stride = max(total // max(sample_k, 1), 1)

    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if stride == 1 or (i % stride == 0 and len(frames) < sample_k):
            frames.append(f)
        i += 1
        if len(frames) >= sample_k:
            break
    return frames

def classify_video_path(video_path, clip_model, labels, text_stack,
                        sample_k=16, use_yolo=True, cropper=None,
                        tau_other=0.27, agg="mean"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames_bgr = sample_frames(cap, sample_k=sample_k)
    cap.release()
    if len(frames_bgr) == 0:
        return "Other", 0.0

    if use_yolo and cropper is not None:
        frames_bgr = [cropper.crop(f) for f in frames_bgr]

    pil_list = [_bgr_to_pil(f) for f in frames_bgr]
    img_emb = clip_model.encode_images(pil_list)   # [K, D]
    label, score, sims = classify_clip_emb(img_emb, text_stack, labels,
                                           agg=agg, tau_other=tau_other)
    return label, score, sims


def evaluate_folder(
    folder_path,
    clip_model,
    text_stack,
    classes,
    labels=None,
    output_dir="outputs",
    sample_k=16,
    use_yolo=True,
    cropper=None,
    tau_other=0.27,
    agg="mean",
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    all_preds, all_labels = [], []
    log_file = os.path.join(output_dir, "predictions.csv")

    def normalize_label(name):
        return name.replace("_", "").upper()

    label_list = labels or list(classes.keys())

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Ground Truth", "Prediction", "Score"])

        for root, dirs, files in os.walk(folder_path):
            # skip the root folder itself
            if root == folder_path:
                continue
            # get ground truth label from folder name
            folder_name = os.path.basename(root)
            gt_label = normalize_label(folder_name)
            if gt_label not in classes:
                continue  # skip folders not in CLASSES

            for fname in files:
                if not fname.endswith(".mp4"):
                    continue

                video_path = os.path.join(root, fname)
                print(f"Processing {video_path}...")

                # --- call the correct function ---
                pred_label, score, sims = classify_video_path(
                    video_path,
                    clip_model,
                    label_list,
                    text_stack,
                    sample_k=sample_k,
                    use_yolo=use_yolo,
                    cropper=cropper,
                    tau_other=tau_other,
                    agg=agg,
                )

                all_preds.append(pred_label)
                all_labels.append(gt_label)

                writer.writerow([fname, gt_label, pred_label, f"{score:.3f}"])

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds, labels=label_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix - Gesture Classification")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.show()

    # --- Accuracy ---
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Overall Accuracy: {acc:.2f}")

    return acc, cm
