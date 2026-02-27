#!/usr/bin/env python3
"""
preprocess.py  (aspect-safe)
Aggressive background removal for fertilizer particle images,
while keeping geometry correct (no aspect squeezing).

Key updates:
- Proportional resize via --resize_width and/or --resize_long
- Optional square letterbox padding via --square_pad
- ROI and exclude polygons auto-rescaled after resize
- Optional undistort using provided intrinsics/distortion

Requires: pip install opencv-python numpy
"""
#parameters selected (adjust it if needed
"args": [
                  "--in_dir", "frames_6s",
                  "--out_dir", "frames_process_6s",
                  "--pattern", "*.png",
                  "--resize_long", "2208",
                  "--square_pad", "2208",
                  "--clahe",
                  "--threshold","otsu",
                  "--top_hat_ksize", "5",
                  "--min_area", "25",
                  "--max_area", "23000",
                  "--min_circ", "0.25",
                  "--max_ar", "2.2",
                  "--max_fill_ratio", "0.75",
                  "--remove_top_k", "3",
                  "--open_ksize", "3",
                  "--close_ksize", "3",
                  "--gamma","0.6"
                ]
                
##############################
import argparse
import fnmatch
import os
import sys
from math import pi
from typing import List, Tuple

import numpy as np

try:
    import cv2
except Exception:
    print("ERROR: OpenCV (cv2) is required. Install with:  pip install opencv-python")
    sys.exit(1)

# -------------------- Args --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Strict particle-only preprocessor (white particles on black)")
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pattern", default="*.png")

    # Proportional resize
    p.add_argument("--resize_width", type=int, default=0, help="Resize to this width, keep aspect (0=skip)")
    p.add_argument("--resize_long", type=int, default=0, help="Resize longest side to this, keep aspect (0=skip)")

    # Optional square letterbox padding (no stretch)
    p.add_argument("--square_pad", type=int, default=0, help="Pad to N×N after processing (0=skip). No stretching.")

    # Tone/contrast & threshold
    p.add_argument("--gamma", type=float, default=1.0, help="<1 brightens before threshold")
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--threshold", type=str, default="adaptive", help="adaptive|otsu|INT")
    p.add_argument("--top_hat_ksize", type=int, default=9, help="Odd size for white top-hat")

    # Include ROI (keep) and exclude ROIs (format: x1,y1,x2,y2,...)
    p.add_argument("--roi", type=str, default="", help="Include polygon 'x1,y1,...' (keep only inside)")
    p.add_argument("--exclude_roi", action="append", default=[], help="Exclude polygon; may repeat")

    # Blob filters
    p.add_argument("--min_area", type=int, default=5)
    p.add_argument("--max_area", type=int, default=5000)
    p.add_argument("--min_circ", type=float, default=0.25)
    p.add_argument("--max_ar", type=float, default=3.0)
    p.add_argument("--max_fill_ratio", type=float, default=0.85, help="area/bbox_area threshold to drop")
    p.add_argument("--remove_top_k", type=int, default=0, help="Remove K largest components/image")

    # Morphology
    p.add_argument("--open_ksize", type=int, default=3)
    p.add_argument("--close_ksize", type=int, default=3)

    # Background reference (optional)
    p.add_argument("--bg_ref", type=str, default="", help="Background reference image to subtract before threshold")

    # Optional undistort (supply full intrinsics)
    p.add_argument("--undistort", action="store_true", help="Undistort input using the given intrinsics/distortion")
    p.add_argument("--fx", type=float, default=0.0)
    p.add_argument("--fy", type=float, default=0.0)
    p.add_argument("--cx", type=float, default=0.0)
    p.add_argument("--cy", type=float, default=0.0)
    p.add_argument("--k1", type=float, default=0.0)
    p.add_argument("--k2", type=float, default=0.0)
    p.add_argument("--p1", type=float, default=0.0)
    p.add_argument("--p2", type=float, default=0.0)
    p.add_argument("--k3", type=float, default=0.0)
    # If your ZED uses higher-order terms, you could extend to k4..k6 here.

    # Misc
    p.add_argument("--preview", action="store_true")
    p.add_argument("--recurse", action="store_true", default=True)
    p.add_argument("--no-recurse", dest="recurse", action="store_false")
    return p.parse_args()



#Camera parameters for ZED2 camera (adjust it for you own)
fx_val=1058.27
fy_val=1057.97
cx_val=1148.22
cy_val=616.90
k1_val=-0.0398
k2_val=0.0078
p1_val=-0.0005
p2_val=-0.0003
k3_val=-0.0042
Xc=0.0
Yc=0.0
Zc=1.2
# Intrinsics (px) and distortion for our *validation* camera
K_val = np.array([[fx_val, 0, cx_val],
                  [0, fy_val, cy_val],
                  [0,      0,      1]], dtype=np.float32)
D_val = np.array([k1_val, k2_val, p1_val, p2_val, k3_val], dtype=np.float32)

# -------------------- Utils --------------------
def list_images(root: str, pattern: str, recurse: bool) -> List[str]:     #get the list of the images
    out = []
    if recurse:
        for r, _d, files in os.walk(root):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    out.append(os.path.join(r, f))
    else:
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p) and fnmatch.fnmatch(f, pattern):
                out.append(p)
    out.sort()
    return out

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img

def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        return gray
    inv = 1.0 / max(1e-6, gamma)
    lut = (np.arange(256) / 255.0) ** inv * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(gray, lut)

def apply_clahe(gray: np.ndarray, use: bool) -> np.ndarray:
    if not use: return gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def parse_poly(poly: str):
    if not poly: return None
    pts = [int(v) for v in poly.split(",")]
    if len(pts) < 6 or len(pts) % 2 != 0:
        raise ValueError("Polygon must be 'x1,y1,x2,y2,...' with >=3 points")
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    return pts

def scale_poly(pts: np.ndarray, sx: float, sy: float):
    if pts is None: return None
    out = pts.copy()
    out[:,0] *= sx
    out[:,1] *= sy
    return out

def polygon_mask(shape_hw: Tuple[int, int], pts: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    if pts is None:
        return np.ones((h, w), dtype=np.uint8) * 255
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask

def threshold_image(gray: np.ndarray, method: str) -> np.ndarray:
    m = method.lower()
    if m == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 5)#31 5
    if m == "otsu":
        _t, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #th[gray < 50] = 0
        return th
    try:
        t = int(method)
        t = max(0, min(255, t))
        _t, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        return th
    except Exception:
        _t, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return th

def white_tophat(gray: np.ndarray, ksize: int) -> np.ndarray:
    k = max(1, ksize) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def undistort_frame(frame_bgr, K, D):
    """Undistort with OpenCV. Returns undistorted BGR and the effective K' used post-rectify. to remove the lens distortion"""
    h, w = frame_bgr.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)  # crop to valid
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)
    undist = cv2.remap(frame_bgr, map1, map2, cv2.INTER_LINEAR)
    return undist, new_K, map1,map2

# -------------------- Core --------------------
def filter_components(bin_img: np.ndarray, args) -> np.ndarray:
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    # Remove K largest components (excluding background)
    if args.remove_top_k > 0 and num > 1:
        areas = stats[1:, 4]
        order = np.argsort(-areas) + 1
        for li in order[:args.remove_top_k]:
            labels[labels == li] = 0

    keep = np.zeros_like(bin_img)
    for label in range(1, num):
        x, y, w, h, area = stats[label]
        if area < args.min_area or area > args.max_area:
            continue

        bbox_area = max(w * h, 1)
        fill_ratio = float(area) / float(bbox_area)
        if fill_ratio > args.max_fill_ratio:
            continue

        mask = (labels == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = cnts[0]

        perim = max(cv2.arcLength(cnt, True), 1e-6)
        circ = 4.0 * pi * (area / (perim * perim))
        if circ < args.min_circ:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), _angle = rect
        rw, rh = max(rw, 1e-6), max(rh, 1e-6)
        ar = max(rw, rh) / max(min(rw, rh), 1e-6)
        if ar > args.max_ar:
            continue

        keep[labels == label] = 255

    return keep

def proportional_resize(img: np.ndarray, resize_width: int, resize_long: int):
    h, w = img.shape[:2]
    if resize_width and resize_long:
        raise ValueError("Use only one of --resize_width or --resize_long.")
    if resize_width and resize_width > 0:
        new_w = resize_width
        new_h = int(round(h * (new_w / float(w))))
        img2 = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sx, sy = new_w / w, new_h / h
        return img2, sx, sy
    if resize_long and resize_long > 0:
        if w >= h:
            new_w = resize_long
            new_h = int(round(h * (new_w / float(w))))
        else:
            new_h = resize_long
            new_w = int(round(w * (new_h / float(h))))
        img2 = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sx, sy = new_w / w, new_h / h
        return img2, sx, sy
    # no resize
    return img, 1.0, 1.0

def letterbox_square(img: np.ndarray, size: int) -> np.ndarray:
    if size <= 0: return img
    h, w = img.shape[:2]
    top = (size - h) // 2 if h < size else 0
    bottom = size - h - top if h < size else 0
    left = (size - w) // 2 if w < size else 0
    right = size - w - left if w < size else 0
    if any(v > 0 for v in (top, bottom, left, right)):
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    # If larger than size (shouldn't happen with this pipeline), center-crop.
    if img.shape[0] > size or img.shape[1] > size:
        y0 = img.shape[0]//2 - size//2
        x0 = img.shape[1]//2 - size//2
        img = img[y0:y0+size, x0:x0+size]
    return img

def process_one(path: str, args, cache: dict) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    
    img_undistort, new_k,map1,map2 = undistort_frame(img,K_val, D_val)

    # Proportional resize (no aspect stretch)
    img_resized, sx, sy = proportional_resize(img_undistort, args.resize_width, args.resize_long)

    gray = to_gray(img_resized)
    gray = apply_gamma(gray, args.gamma)
    gray = apply_clahe(gray, args.clahe)

    # Background reference subtraction
    if args.bg_ref:
        bg = cache.get("bg")
        if bg is None:
            bg_img = cv2.imread(args.bg_ref, cv2.IMREAD_GRAYSCALE)
            if bg_img is None:
                raise RuntimeError(f"Failed to read bg_ref: {args.bg_ref}")
            if bg_img.shape != gray.shape:
                bg_img = cv2.resize(bg_img, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
            cache["bg"] = bg_img
            bg = bg_img
        gray = cv2.absdiff(gray, bg)

    th_src = white_tophat(gray, args.top_hat_ksize)
    bin0 = threshold_image(th_src, args.threshold)

    # Build masks with auto-scaled polygons
    h, w = bin0.shape[:2]
    key = (w, h, sx, sy, args.roi, tuple(args.exclude_roi))

    if "masks_key" not in cache or cache["masks_key"] != key:
        base_w = cache.get("base_w", w / sx)
        base_h = cache.get("base_h", h / sy)
        cache["base_w"] = base_w
        cache["base_h"] = base_h

        inc_pts = parse_poly(args.roi)
        exc_pts_list = [parse_poly(p) for p in args.exclude_roi] if args.exclude_roi else []

        # scale polygons from original coordinates to current image size
        sxx = w / base_w if base_w else 1.0
        syy = h / base_h if base_h else 1.0

        inc_pts_s = scale_poly(inc_pts, sxx, syy) if inc_pts is not None else None
        exc_pts_s = [scale_poly(p, sxx, syy) for p in exc_pts_list if p is not None]

        inc_mask = polygon_mask((h, w), inc_pts_s)
        cache["inc_mask"] = inc_mask
        cache["exc_masks"] = [polygon_mask((h, w), p) for p in exc_pts_s]
        cache["masks_key"] = key

    bin0 = cv2.bitwise_and(bin0, cache["inc_mask"])
    for m in cache["exc_masks"]:
        bin0 = cv2.bitwise_and(bin0, cv2.bitwise_not(m))

    # Morphology
    if args.open_ksize and args.open_ksize > 0:
        k = max(1, args.open_ksize) | 1
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bin0 = cv2.morphologyEx(bin0, cv2.MORPH_OPEN, open_k, iterations=1)
    if args.close_ksize and args.close_ksize > 0:
        k = max(1, args.close_ksize) | 1
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bin0 = cv2.morphologyEx(bin0, cv2.MORPH_CLOSE, close_k, iterations=1)

    #out = filter_components(bin0, args)

    # Optional square letterbox (no geometry distortion)
    if args.square_pad and args.square_pad > 0:
        out = letterbox_square(bin0, args.square_pad) #out

    return (out > 0).astype(np.uint8) * 255

# -------------------- Main --------------------
def main():
    args = parse_args()
    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(in_dir):
        print(f"ERROR: input directory not found: {in_dir}")
        sys.exit(1)
    ensure_dir(out_dir)

    if args.resize_width and args.resize_long:
        print("ERROR: Use only one of --resize_width or --resize_long.")
        sys.exit(1)

    imgs = list_images(in_dir, args.pattern, args.recurse)
    if not imgs:
        print(f"No images matched {args.pattern} under {in_dir}")
        sys.exit(1)
    print(f"Found {len(imgs)} images. Strict particle-only filtering...")

    cache = {}
    for i, src in enumerate(imgs, 1):
        try:
            out = process_one(src, args, cache)
            rel = os.path.relpath(src, in_dir)
            dst = os.path.join(out_dir, os.path.splitext(rel)[0] + ".png")
            ensure_dir(os.path.dirname(dst))
            cv2.imwrite(dst, out)
        except Exception as e:
            print(f"[WARN] {src}: {e}")

        if args.preview:
            cv2.imshow("strict", out)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        if i % 50 == 0 or i == len(imgs):
            print(f"Processed {i}/{len(imgs)}")

    if args.preview:
        cv2.destroyAllWindows()
    print(f"Done. Outputs in: {out_dir}")

if __name__ == "__main__":
    main()


