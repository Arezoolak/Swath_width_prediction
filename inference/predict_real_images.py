#!/usr/bin/env python3

import os
import glob
import argparse
import cv2
import numpy as np
import torch
import natsort

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from feature_extractor import CNNFeatureExtractor
from transformer import SwathWidthTransformer


# -----------------------
# ARGUMENT PARSER
# -----------------------
parser = argparse.ArgumentParser(description="Swath width prediction from frame sequence")

parser.add_argument('--frames_dir', type=str, required=True,
                    help='Directory containing sequential frames')

parser.add_argument('--path_weights', type=str, required=True,
                    help='Path to trained model weights')

parser.add_argument('--last_frame', type=str, required=True,
                    help='Frame used for visualization')

args = parser.parse_args()

frames_dir = args.frames_dir
weights_path = args.path_weights
last_frame_path = args.last_frame


# -----------------------
# CAMERA CALIBRATION
# -----------------------
fx_val=1058.27
fy_val=1057.97
cx_val=1148.22
cy_val=616.90

k1_val=-0.0398
k2_val=0.0078
p1_val=-0.0005
p2_val=-0.0003
k3_val=-0.0042

K_val = np.array([
    [fx_val, 0, cx_val],
    [0, fy_val, cy_val],
    [0, 0, 1]
], dtype=np.float32)

D_val = np.array([k1_val, k2_val, p1_val, p2_val, k3_val], dtype=np.float32)


# Camera pose
Xc=0.0
Yc=0.0
Zc=1.2

cam_pos_w = np.array([Xc, Yc, Zc], dtype=np.float32)

yaw_deg = 0.0
pitch_deg = -30.0
roll_deg = 0.0


# -----------------------
# ORTHO GRID SETTINGS
# -----------------------
ORTHO_W_M = 60.0
ORTHO_H_M = 60.0

ORTHO_W_PX = 512
ORTHO_H_PX = 512

M_PER_PX = ORTHO_W_M / ORTHO_W_PX


# -----------------------
# HELPER FUNCTIONS
# -----------------------

def euler_to_R(yaw, pitch, roll):

    rz = np.deg2rad(yaw)
    rx = np.deg2rad(pitch)
    ry = np.deg2rad(roll)

    Rz = np.array([
        [np.cos(rz),-np.sin(rz),0],
        [np.sin(rz),np.cos(rz),0],
        [0,0,1]
    ])

    Rx = np.array([
        [1,0,0],
        [0,np.cos(rx),-np.sin(rx)],
        [0,np.sin(rx),np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry),0,np.sin(ry)],
        [0,1,0],
        [-np.sin(ry),0,np.cos(ry)]
    ])

    return Rz @ Rx @ Ry


def undistort_frame(frame, K, D):

    h,w = frame.shape[:2]

    new_K,_ = cv2.getOptimalNewCameraMatrix(K,D,(w,h),0)

    map1,map2 = cv2.initUndistortRectifyMap(
        K,D,None,new_K,(w,h),cv2.CV_16SC2)

    undist = cv2.remap(frame,map1,map2,cv2.INTER_LINEAR)

    return undist,new_K


def img_to_grid_h_from_extrinsics(K_rect,R_wc,t_wc):

    H_g2i = K_rect @ np.c_[R_wc[:,0],R_wc[:,1],t_wc]

    H_i2g = np.linalg.inv(H_g2i)

    S = 1.0 / M_PER_PX

    A = np.array([
        [S,0,ORTHO_W_PX*0.5],
        [0,-S,ORTHO_H_PX*0.5],
        [0,0,1]
    ])

    return A @ H_i2g


def warp_to_ortho(frame,H):

    return cv2.warpPerspective(
        frame,H,(ORTHO_W_PX,ORTHO_H_PX),
        flags=cv2.INTER_LINEAR
    )


# -----------------------
# MODEL CONFIG
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_frames = 25
backbone = "resnet18"


# -----------------------
# TRANSFORMS
# -----------------------
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])


# -----------------------
# LOAD MODELS
# -----------------------
cnn = CNNFeatureExtractor(backbone=backbone, pretrained=True).to(device).eval()

model = SwathWidthTransformer(
    feature_dim=512,
    num_frames=num_frames
).to(device)

model.load_state_dict(torch.load(weights_path,map_location=device))
model.eval()


# -----------------------
# LOAD FRAMES
# -----------------------
frame_paths = natsort.natsorted(
    glob.glob(os.path.join(frames_dir,"*.png")) +
    glob.glob(os.path.join(frames_dir,"*.jpg"))
)

if len(frame_paths)==0:
    raise RuntimeError("No frames found")

if len(frame_paths)<num_frames:
    frame_paths += [frame_paths[-1]]*(num_frames-len(frame_paths))
else:
    frame_paths = frame_paths[:num_frames]


# -----------------------
# BUILD HOMOGRAPHY
# -----------------------
sample = cv2.imread(frame_paths[0])

undist,K_rect = undistort_frame(sample,K_val,D_val)

R_wc = euler_to_R(yaw_deg,pitch_deg,roll_deg)

t_wc = -R_wc @ cam_pos_w.reshape(3,1)
t_wc = t_wc.reshape(3)

H_i2grid = img_to_grid_h_from_extrinsics(K_rect,R_wc,t_wc)


# -----------------------
# FEATURE EXTRACTION
# -----------------------
features=[]

with torch.no_grad():

    for p in frame_paths:

        img=cv2.imread(p)

        undist,_ = undistort_frame(img,K_val,D_val)

        ortho = warp_to_ortho(undist,H_i2grid)

        rgb = cv2.cvtColor(ortho,cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)

        x = transform(pil).unsqueeze(0).to(device)

        f = cnn(x)

        features.append(f.squeeze(0))


features = torch.stack(features).unsqueeze(0)


# -----------------------
# PREDICTION
# -----------------------
with torch.no_grad():

    pred_width_m = model(features).item()

print(f"Predicted swath width: {pred_width_m:.2f} m")


# -----------------------
# VISUALIZATION
# -----------------------
base_img = Image.open(last_frame_path).convert("RGB")

draw = ImageDraw.Draw(base_img)

font_size = max(22, base_img.width // 32)

try:
    font = ImageFont.truetype("DejaVuSans.ttf",font_size)
except:
    font = ImageFont.load_default()

text = f"Predicted swath width: {pred_width_m:.2f} m"

bbox = draw.textbbox((0,0),text,font=font)

text_w = bbox[2]-bbox[0]
text_h = bbox[3]-bbox[1]

pad = 10

x = pad
y = base_img.height-text_h-2*pad

draw.rectangle([x-pad,y-pad,x+text_w+pad,y+text_h+pad],fill=(0,0,0))

draw.text((x,y),text,(255,255,255),font=font)


out_path = os.path.join(os.path.dirname(last_frame_path),"prediction.png")

base_img.save(out_path)

print("Saved result to:",out_path)

