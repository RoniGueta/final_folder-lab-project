import os
import timeit
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from functools import lru_cache
from numpy import pi
from cv2 import ORB_create, ORB_FAST_SCORE
from pathlib import Path


cloud = np.empty((0, 3))
image = np.full((200, 200, 3), 255, dtype=np.uint8)

def load_camera_data_json(json_path, dtype=np.float32):
    with open(json_path, "r") as file:
        data = json.load(file)
    return np.array(data['k'], dtype=dtype), np.array(data['d'], dtype=dtype), np.array(data['dims'], dtype=np.int32)

def depth_from_h264_vectors(vectors: np.ndarray, camera_matrix: np.ndarray, diff: float):
    """
    Estimates depth of each 3d point, represented as a vector(x1, y1, x2, y2) between its
    2d location in 2 consecutive frames with a known height difference
    """
    assert diff != 0
    if len(vectors) == 0:
        return np.empty(0)
    with np.errstate(divide='ignore'):
        return diff * camera_matrix[1, 2] / np.abs(vectors[:, 3] - vectors[:, 1])
@lru_cache(maxsize=1)
def generate_end_block_matrix(shape) -> np.ndarray:
    return np.fromfunction(lambda h, w, d: (8 + 16 * h) * d + (8 + 16 * w) * (1 - d), (shape[0], shape[1], 2),
                           dtype=np.int16)
def pi_vectors_conversion(mv_vectors: np.ndarray) -> np.ndarray:
    end_matrix: np.ndarray = generate_end_block_matrix(mv_vectors.shape)
    if mv_vectors is None or mv_vectors.shape == ():
        return np.empty((0, 0))
    to_keep = (mv_vectors['x'] != 0) | (mv_vectors['y'] != 0)
    combined = np.empty((np.sum(to_keep), 4), dtype=np.float32)
    end_matrix = end_matrix[to_keep]
    mv_vectors = mv_vectors[to_keep]
    combined[:, 0] = end_matrix[:, 0] + mv_vectors['x']
    combined[:, 1] = end_matrix[:, 1] + mv_vectors['y']
    combined[:, 2:] = end_matrix
    return combined
def topdown_view(depth: np.ndarray, angle: float, max_dist: float = 1500):
    global image
    # depth map to cloud, clip it at max_dist to prevent extremely far outliers
    depth[:, 2] = np.clip(depth[:, 2], 0, max_dist)

    depth[:, :2] = np.squeeze(cv2.undistortPoints(depth[None, :, :2], cam_mat, dist_coeff)) * depth[:, 2:]
    # depth is now a Nx3 3d point cloud
    rot_mat = Rotation.from_euler('y', angle, degrees=True).as_matrix()
    depth = depth @ rot_mat
    for point in depth:
        point = np.clip((point * 200 / max_dist + 100).astype(int), -200, 200)
        cv2.circle(image, (point[0], point[2]), 1, (0, 0, 0), -1)
    return image


def topdown_view(depth: np.ndarray, angle: float, max_dist: float = 1500):
    global image
    # depth map to cloud, clip it at max_dist to prevent extremely far outliers
    depth[:, 2] = np.clip(depth[:, 2], 0, max_dist)

    depth[:, :2] = np.squeeze(cv2.undistortPoints(depth[None, :, :2], cam_mat, dist_coeff)) * depth[:, 2:]
    # depth is now a Nx3 3d point cloud
    rot_mat = Rotation.from_euler('y', angle, degrees=True).as_matrix()
    depth = depth @ rot_mat
    for point in depth:
        point = np.clip((point * 200 / max_dist + 100).astype(int), -200, 200)
        cv2.circle(image, (point[0], point[2]), 1, (0, 0, 0), -1)
    return image

def create_masks_vertical(frame: np.ndarray, motion_vectors: np.ndarray, frame2: np.ndarray = None,
                          overlay_frames: bool = False):
    w = len(frame)
    h = len(frame[0])
    mask1 = np.zeros((w, h), dtype=np.uint8)
    mask2 = np.zeros((w, h), dtype=np.uint8)
    for vector in motion_vectors:
        cv2.rectangle(mask1, (vector[:2]).astype(int) - 8, (vector[:2]).astype(int) + 8, (255, 255, 255), -1)
        if overlay_frames:
            cv2.rectangle(frame, (vector[:2]).astype(int) - 8, (vector[:2]).astype(int) + 8, (255, 255, 255), 1)
        cv2.rectangle(mask2, (vector[2:]).astype(int) - 8, (vector[2:]).astype(int) + 8, (255, 255, 255), -1)
        if overlay_frames:
            cv2.rectangle(frame2, (vector[2:]).astype(int) - 8, (vector[2:]).astype(int) + 8, (255, 255, 255), 1)
    return (frame, frame2, mask1, mask2) if overlay_frames else (mask1, mask2)



def create_masks_horizontal(frame: np.ndarray, post_mv: np.ndarray, pre_mv: np.ndarray, method1: int,
                            overlay_frames: bool = False):
    w = len(frame)//16
    h = len(frame[0])//16
    if method1 == 0:
        mask1 = pre_mv["x"][:, :40].astype(np.uint8)
        mask1[mask1 > np.percentile(mask1, 25)] = 255
        mask1[mask1 <= np.percentile(mask1, 25)] = 0
        return cv2.resize(mask1, (640, 480), interpolation=cv2.INTER_NEAREST)
    elif method1 == 1:
        mask1 = np.zeros((w, h), dtype=np.uint8)
        for i in range(40):
              for j in range(30):
                  if pre_mv[j, i]["x"] != 0:
                      mask1[j, i] = 255
        return cv2.resize(mask1, (640, 480), interpolation=cv2.INTER_NEAREST)
    else:
        w = len(frame)
        h = len(frame[0])
        mask1 = np.zeros((w, h), dtype=np.uint8)
        for vector in post_mv:
            cv2.rectangle(mask1, vector[2:] - 8, vector[2:] + 8, (255, 255, 255), -1)
            if overlay_frames:
                cv2.rectangle(frame, vector[2:] - 8, vector[2:] + 8, (255, 255, 255), 1)
        return (frame, mask1) if overlay_frames else mask1


save_video: bool = False  # turn off when saved video not required
show_video: bool = True  # turn off when video window not required
top_down: bool = True  # turn on when working on topdown view
show_depth_frame: bool = True
compete_times: bool = True  # if you want to compete between mask, no mask
signal_type: bool = True  # horizontal for true, vertical false
cam_dir = os.path.join("camera_data_480p.json")
cam_mat, dist_coeff, _ = load_camera_data_json(cam_dir)
path = "depth_test1"
tello_angles = np.loadtxt(os.path.join(path, "tello_angles_low.csv"))

pre_mv1 = np.load(os.path.join(path, "motion_data_low.npy")).reshape((-1, 30, 41))
post_mv1 = [pi_vectors_conversion(mv).astype(int) for mv in pre_mv1]
print(len(post_mv1))
pre_mv2 = np.load(os.path.join(path, "motion_data_high.npy")).reshape((-1, 30, 41))
post_mv2 = [pi_vectors_conversion(mv).astype(int) for mv in pre_mv2]
# angles1 = np.loadtxt(os.path.join(path, "tello_angles1.csv"))
           # angles2 = np.loadtxt(os.path.join(path, "tello_angles2.csv"))
           # best_pair = generate_angle_pairs(angles1, angles2)  # doesn't work
cap1 = cv2.VideoCapture(os.path.join(path, "rot_low.h264"))
cap2 = cv2.VideoCapture(os.path.join(path, "rot_high.h264"))
if save_video:
    writer = cv2.VideoWriter(os.path.join(path, "depth.mp4"), -1, 40, (640, 480))
else:
    writer = None  # just to stop warning
# Initialize the feature detector (e.g., ORB, SIFT, etc.)
i = 0
detector = cv2.ORB_create(nfeatures=1000, nlevels=1, edgeThreshold=4, patchSize=4)
for angle, post1, pre1, post2, pre2 in zip(tello_angles, post_mv1, pre_mv1, post_mv2, pre_mv2):
    ret1, frame1 = cap1.read()
    if not ret1:
        if save_video:
            writer.release()
        break
    ret2, frame2 = cap2.read()
    if compete_times and i < 100:
        i += 1
        continue
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    def func1(mask1, mask2):
        keypoints_1 = detector.detect(gray1, mask1)
        keypoints_2 = detector.detect(gray2, mask2)
        return keypoints_1, keypoints_2
    def func2(pic1, pic2):
        keypoints_1 = detector.detect(pic1, None)
        keypoints_2 = detector.detect(pic2, None)
        return keypoints_1, keypoints_2
    def func3(key1, key2):
        descriptors_1 = detector.compute(gray1, key1)
        descriptors_2 = detector.compute(gray2, key2)
        return descriptors_1, descriptors_2
    def func4(mask1, mask2):
        keypoints_1, descriptors_1 = detector.detectAndCompute(gray1, mask1)
        keypoints_2, descriptors_2 = detector.detectAndCompute(gray1, mask2)
        return keypoints_1, descriptors_1, keypoints_2, descriptors_2
    def func5():
        keypoints_1, descriptors_1 = detector.detectAndCompute(gray1, None)
        keypoints_2, descriptors_2 = detector.detectAndCompute(gray1, None)
        return keypoints_1, descriptors_1, keypoints_2, descriptors_2

    def second_way(post_mv1, post_mv2, block_radius = 16):
        keypoints1_new = []
        keypoints2_new = []
        for mv_dot1 in post_mv1:
            image1 = gray1[max(0, mv_dot1[3]-block_radius):(mv_dot1[3]+block_radius), max(0, mv_dot1[2]-block_radius):(mv_dot1[2]+block_radius)]

            key_points1_second_way = detector.detect(image1, None)

            modified_keypoints1 = [
                (pt.pt[0] + max(0, mv_dot1[3]-block_radius), pt.pt[1] + max(0, mv_dot1[2]-block_radius))
                for pt in key_points1_second_way
            ]
            keypoints1_new.extend(modified_keypoints1)
        for mv_dot2 in post_mv2:
            image2 = gray2[max(0, mv_dot2[3]-block_radius):(mv_dot2[3]+block_radius), max(0, mv_dot2[2]-block_radius):(mv_dot2[2]+block_radius)]
            key_points2_second_way = detector.detect(image2, None)
            modified_keypoints2 = [
                (pt.pt[0] + mv_dot2[3] - block_radius, pt.pt[1] + mv_dot2[2] - block_radius)
                for pt in key_points2_second_way
            ]
            keypoints2_new.extend(modified_keypoints2)
        keypoints1_new = tuple(keypoints1_new)
        keypoints2_new = tuple(keypoints2_new)
        return keypoints1_new, keypoints2_new


    if signal_type:  # herizontal
        def check_diff_algo1():
            print(timeit.timeit(lambda:  create_masks_horizontal(frame1, post1, pre1, 0), number=1000))
            print(timeit.timeit(lambda:  create_masks_horizontal(frame1, post1, pre1, 1), number=1000))
            print(timeit.timeit(lambda:  create_masks_horizontal(frame1, post1, pre1, 2), number=1000))
            exit()
        # check_diff_algo1()
        mask1 = create_masks_horizontal(frame1, post1, pre1, 0)   # Use this line instead of the one above when not showing below
        mask2 = create_masks_horizontal(frame2, post2, pre2, 0)   # Use this line instead of the one above when not showing below
        # frame1_1, mask1 = create_masks_horizontal(frame1, post1, pre1, 2, True)
        # frame2_2, mask2 = create_masks_horizontal(frame2, post2, pre2, 2, True)
    else:  # vertical
        vectors = np.loadtxt("vectors.csv")
        def check_diff_algo2():
            print(timeit.timeit(lambda:  create_masks_vertical(frame1, vectors, frame2, True), number=1000))
            exit()
        # check_diff_algo2()
        frame_1, frame_2, mask1, mask2 = create_masks_vertical(frame1, vectors, frame2, True)
        # mask1, mask2 = create_masks_vertical(frame1_vert, vectors, frame2_vert, False)
    if compete_times:
        if signal_type:
            print("horizontal:")
        else:
            print("vertical:")
        print("with mask: ")
        print(timeit.timeit(lambda: func4(mask1, mask2), number=1000))
        print("     detect: ", timeit.timeit(lambda: func1(mask1, mask2), number=1000))
        key_1_mask, key_2_mask = func1(mask1, mask2)
        print("     compute: ", timeit.timeit(lambda: func3(key_1_mask, key_2_mask), number=1000))
        print("without mask: ")
        print(timeit.timeit(lambda: func5(), number=1000))
        print("     detect:", timeit.timeit(lambda: func2(gray1, gray2), number=1000))
        key_1_without, key_2_without = func2(gray1, gray2)
        print("     compute: ", timeit.timeit(lambda: func3(key_1_without, key_2_without), number=1000))
        print("with new way: ")

        #first way
        new_gray1 = gray1.copy()
        new_gray2 = gray2.copy()
        new_gray1[mask1 == 0] = 0
        new_gray2[mask2 == 0] = 0
        #

        print("     detect:", timeit.timeit(lambda: func2(new_gray1, new_gray2), number=1000))
        key_1_first_way, key_2_first_way = func2(new_gray1, new_gray2)
        print("second new way:")
        print("     detect:", timeit.timeit(lambda: second_way(post1, post2), number=1000))
        key_1_second_way, key_2_second_way = second_way(post1, post2)
        print(len(key_1_mask), len(key_1_without), len(key_1_second_way))

        def show_keypoints(keys, direct_tuples = False):
            image = frame1.copy()
            for point in keys:
                if direct_tuples:
                    cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 0, 0), -1)
                else:
                    cv2.circle(image, (int(point.pt[0]), int(point.pt[1])), 1, (0, 0, 0), -1)
            return image
        cv2.imshow("with mask", show_keypoints(key_1_mask))
        cv2.imshow("without mask", show_keypoints(key_1_without))
        cv2.imshow("first way", show_keypoints(key_1_first_way))
        cv2.imshow("second way", show_keypoints(key_1_second_way, True))
        cv2.imshow("mask1", mask1)
        cv2.waitKey()
        input()
        exit()
    cv2.imshow("mask1", mask1)
    cv2.imshow("mask2", mask2)
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, mask1)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Perform the matching
    matches = matcher.match(descriptors1, descriptors2)
    print(len(matches), "matches using ORB")
    # show matches

    # frame_with_matches = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, matches[:100], None)
    # cv2.imshow("ORB matches", frame_with_matches)
    # cv2.waitKey(0)
    # continue
    # show depth
    points1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
    points2 = np.array([keypoints2[match.trainIdx].pt for match in matches])
    depth = depth_from_h264_vectors(np.hstack((points1, points2)), cam_mat,
                                    30)  # you might want to save one of these for the topdown view
    if top_down and len(depth) != 0:
        top_down_frame = topdown_view(np.hstack((points1, depth[:, None])), angle)
        if show_video:
            cv2.imshow("depth top down", top_down_frame)
    if show_depth_frame:
        depth_frame = frame1.copy()
        int_points1 = points1.astype(int)
        depth_color = np.clip(depth * 255 / 500, 0, 255)[:, None]  # clip  values from 0 to 5m and scale to 0-255(color range)
        for color, point in zip(depth_color, int_points1):
            cv2.rectangle(depth_frame, point[::] - 5, point[::] + 5, color, -1)
        if show_video:
            cv2.imshow("depth ORB", depth_frame)
            cv2.waitKey(1)  # need some minimum time because opencv doesnt work without it
    if save_video:
        writer.write(depth_frame)
input()