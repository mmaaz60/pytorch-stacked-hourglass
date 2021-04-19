import os
import cv2
import numpy as np


def draw_keypoints_on_image(folder_path, image_path, keypoints, index=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    keypoints = keypoints.detach()
    keypoints = np.array(keypoints).reshape(16, -1)
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    for i, joint in enumerate(keypoints):
        joint_x = int(joint[0])
        joint_y = int(joint[1])
        if index is not None and index != i:
            continue
        if joint_x < w and joint_y < h:
            cv2.circle(image, (joint_x, joint_y), radius=0, color=(0, 0, 255), thickness=5)
    cv2.imwrite(f"{folder_path}/keypoints_{image_path.split('/')[-1]}", image)


def draw_skeleton_on_image(folder_path, image_path, keypoints):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    keypoints = keypoints.detach()
    keypoints = np.array(keypoints).reshape(16, -1)
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    joints = []
    for i, joint in enumerate(keypoints):
        joint_x = int(joint[0])
        joint_y = int(joint[1])
        joints.append((joint_x, joint_y))
    # draw skeleton
    for i, bone in enumerate(MPII_BONES):
        joint_1 = joints[bone[0]]
        joint_2 = joints[bone[1]]
        cv2.line(image, joint_1, joint_2, thickness=5, color=(0 + (i+5)*10, 0 + (i+2)*5, 255 - (i+3)*10))
    for joint in joints:
        cv2.circle(image, (joint[0], joint[1]), radius=0, color=(0, 255, 0), thickness=5)
    cv2.imwrite(f"{folder_path}/skeleton_{image_path.split('/')[-1]}", image)


R_ANKLE = 0
R_KNEE = 1
R_HIP = 2
L_HIP = 3
L_KNEE = 4
L_ANKLE = 5
PELVIS = 6
THORAX = 7
UPPER_NECK = 8
HEAD_TOP = 9
R_WRIST = 10
R_ELBOW = 11
R_SHOULDER = 12
L_SHOULDER = 13
L_ELBOW = 14
L_WRIST = 15

MPII_BONES = [
    [R_ANKLE, R_KNEE],
    [R_KNEE, R_HIP],
    [R_HIP, PELVIS],
    [L_HIP, PELVIS],
    [L_HIP, L_KNEE],
    [L_KNEE, L_ANKLE],
    [PELVIS, THORAX],
    [THORAX, UPPER_NECK],
    [UPPER_NECK, HEAD_TOP],
    [R_WRIST, R_ELBOW],
    [R_ELBOW, R_SHOULDER],
    [THORAX, R_SHOULDER],
    [THORAX, L_SHOULDER],
    [L_SHOULDER, L_ELBOW],
    [L_ELBOW, L_WRIST]
]
