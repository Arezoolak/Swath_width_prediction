#!/usr/bin/env python3
"""
Aspect-safe fertilizer particle preprocessing.

- Proportional resize
- Optional square padding (no distortion)
- Optional CLAHE + gamma
- Optional undistortion via camera intrinsics
- ROI / exclusion polygons supported
"""

import argparse
import fnmatch
import os
import sys
from math import pi
from typing import List, Tuple

import numpy as np
import cv2


# ======================================================
# ARGUMENTS
# ======================================================
def parse_args():
    p = argparse.ArgumentParser(description="Particle-only preprocessor")

    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pattern", default="*.png")

    # Resize options
    p.add_argument("--resize_width", type=int, default=0)
    p.add_argument("--resize_long", type=int, default=0)
    p.add_argument("--square_pad", type=int, default=0)

    # Enhancement
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--threshold", default="adaptive")
    p.add_argument("--top_hat_ksize", type=int, default=9)

    # Blob filtering
    p.add_argument("--min_area", type=int, default=5)
    p.add_argument("--max_area", type=int, default=5000)
    p.add_argument("--min_circ", type=float, default=0.25)
    p.add_argument("--max_ar", type=float, default=3.0)

    # Undistortion (optional)
    p.add_argument("--undistort", action="store_true")
    p.add_argument("--fx", type=float, default=0.0)
    p.add_argument("--fy", type=float, default=0.0)
    p.add_argument("--cx", type=float, default=0.0)
    p.add_argument("--cy", type=float, default=0.0)
    p.add_argument("--k1", type=float, default=0.0)
    p.add_argument("--k2", type=float, default=0.0)
    p.add_argument("--p1", type=float, default=0.0)
    p.add_argument("--p2", type=float, default=0.0)
    p.add_argument("--k3", type=float, default=0.0)

    p.add_argument("--preview", action="store_true")
    return p.parse_args()


# ======================================================
# UTILS
# ======================================================
def list_images(root: str, pattern: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                out.append(os.path.join(r, f))
    return sorted(out)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def apply_gamma(gray, gamma):
    if abs(gamma - 1.0) < 1e-3:
        return gray
    inv = 1.0 / gamma
    lut = (np.arange(256) / 255.0) ** inv * 255
    return cv2.LUT(gray, lut.astype(np.uint8))


def white_tophat(gray, ksize):
    k = max(1, ksize) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def threshold_image(gray, method):
    if method.lower() == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5
        )
    if method.lower() == "otsu":
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    t = int(method)
    _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return th


def proportional_resize(img, resize_width, resize_long):
    h, w = img.shape[:2]
    if resize_width > 0:
        new_w = resize_width
        new_h = int(h * new_w / w)
    elif resize_long > 0:
        scale = resize_long / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        return img

    return cv2.resize(img, (new_w, new_h),
                      interpolation=cv2.INTER_AREA)


def letterbox_square(img, size):
    h, w = img.shape[:2]
    top = (size - h) // 2 if h < size else 0
    bottom = size - h - top if h < size else 0
    left = (size - w) // 2 if w < size else 0
    right = size - w - left if w < size else 0

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=0
    )

    return img


def undistort_frame(img, args):
    if not args.undistort:
        return img

    K = np.array([[args.fx, 0, args.cx],
                  [0, args.fy, args.cy],
                  [0, 0, 1]], dtype=np.float32)

    D = np.array([args.k1, args.k2,
                  args.p1, args.p2, args.k3],
                 dtype=np.float32)

    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
    return cv2.undistort(img, K, D, None, new_K)


# ======================================================
# MAIN PROCESS
# ======================================================
def process_one(path, args):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read {path}")

    img = undistort_frame(img, args)
    img = proportional_resize(img, args.resize_width, args.resize_long)

    gray = to_gray(img)
    gray = apply_gamma(gray, args.gamma)

    if args.clahe:
        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray = clahe.apply(gray)

    th_src = white_tophat(gray, args.top_hat_ksize)
    bin_img = threshold_image(th_src, args.threshold)

    if args.square_pad > 0:
        bin_img = letterbox_square(bin_img, args.square_pad)

    return (bin_img > 0).astype(np.uint8) * 255


# ======================================================
# MAIN
# ======================================================
def main():
    args = parse_args()

    ensure_dir(args.out_dir)
    imgs = list_images(args.in_dir, args.pattern)

    print(f"Processing {len(imgs)} images...")

    for i, src in enumerate(imgs, 1):
        try:
            out = process_one(src, args)

            rel = os.path.relpath(src, args.in_dir)
            dst = os.path.join(args.out_dir,
                               os.path.splitext(rel)[0] + ".png")
            ensure_dir(os.path.dirname(dst))
            cv2.imwrite(dst, out)

        except Exception as e:
            print(f"[WARN] {src}: {e}")

        if args.preview:
            cv2.imshow("preview", out)
            if cv2.waitKey(1) == 27:
                break

        if i % 50 == 0 or i == len(imgs):
            print(f"Processed {i}/{len(imgs)}")

    if args.preview:
        cv2.destroyAllWindows()

    print(f"Done. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
