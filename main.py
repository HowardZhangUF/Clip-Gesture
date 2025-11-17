# main.py  —  video-only entrypoint

import argparse
import glob
import os
import sys

from config import (
    BACKBONE, PRETRAINED, CLASSES, TAU_OTHER,
    SAMPLE_K, USE_YOLO, YOLO_PAD
)
from clip_model import ClipZeroShot
from classify import evaluate_folder
from smoothing import TemporalSmoother
from detection import PersonCropper
def expand_inputs(patterns):
    paths = []
    for p in patterns:
        if any(ch in p for ch in ["*", "?", "["]):
            paths.extend(sorted(glob.glob(p)))
        else:
            paths.append(p)
    # de-dup + keep order
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def main():
    ap = argparse.ArgumentParser(description="Zero-shot CLIP diver gesture classifier (video files only)")
    ap.add_argument("--video", "-v", nargs="+", required=True,
                    help="Path(s) or glob(s) to video(s). Example: data/*.mp4")
    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "median", "max"],
                    help="Temporal aggregation over frames")
    ap.add_argument("--no_yolo", action="store_true",
                    help="Disable YOLO person crop; use full frame")
    ap.add_argument("--print_scores", action="store_true",
                    help="Print per-class cosine similarity scores")
    ap.add_argument("--k", type=int, default=SAMPLE_K,
                    help=f"Frames sampled per video clip (default {SAMPLE_K})")
    args = ap.parse_args()

    # Resolve inputs
    video_paths = expand_inputs(args.video)
    if not video_paths:
        print("No videos matched the provided paths/patterns.", file=sys.stderr)
        sys.exit(1)

    # Load CLIP + text prototypes once
    clip = ClipZeroShot(BACKBONE, PRETRAINED)
    labels, text_stack = clip.build_text_prototypes(CLASSES)

    # Optional YOLO cropper
    cropper = None
    if USE_YOLO and not args.no_yolo:
        cropper = PersonCropper(pad=YOLO_PAD)

    # Process each video
    ok_count, fail_count = 0, 0
    for vp in video_paths:
        try:
            label, score, sims = ClipZeroShot(
                video_path=vp,
                clip_model=clip,
                labels=labels,
                text_stack=text_stack,
                sample_k=args.k,
                use_yolo=(cropper is not None),
                cropper=cropper,
                tau_other=TAU_OTHER,
                agg=args.agg
            )
            base = os.path.basename(vp)
            print(f"[{base}]  ->  {label} (score={score:.3f})")
            if args.print_scores:
                # sorted by score desc
                order = sorted(range(len(labels)), key=lambda i: float(sims[i]), reverse=True)
                for i in order:
                    print(f"   {labels[i]:<12s}: {float(sims[i]):.3f}")
            ok_count += 1
        except Exception as e:
            print(f"[ERROR] {vp}: {e}", file=sys.stderr)
            fail_count += 1

    # Summary
    print(f"\nDone. Success: {ok_count}  |  Failed: {fail_count}")

if __name__ == "__main__":
    video_folder = "video/"    # folder with test videos
    output_dir = "outputs/"     # save results here

    # Load CLIP
    clip_model = ClipZeroShot(BACKBONE, PRETRAINED)

    # Build text embeddings
    labels, text_embeddings = clip_model.build_text_prototypes(CLASSES)
    # Temporal smoother
    smoother = TemporalSmoother()

    # Run evaluation on all videos
    acc, cm = evaluate_folder(video_folder, clip_model, text_embeddings, CLASSES, output_dir=output_dir)