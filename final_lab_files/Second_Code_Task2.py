import json
from functools import lru_cache
import os
from itertools import repeat
import timeit
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from tqdm import tqdm
from cv2 import ORB_create, ORB_FAST_SCORE
from numpy import pi
from pathlib import Path

# from mapping.utils.file import load_camera_data_json

def load_camera_data_json(json_path, dtype=np.float32):
    with open(json_path, "r") as file:
        data = json.load(file)
    return np.array(data['k'], dtype=dtype), np.array(data['d'], dtype=dtype), np.array(data['dims'], dtype=np.int32)


image = np.full((200, 200, 3), 255, dtype=np.uint8)

save_video: bool = False  # turn off when saved video not required
show_video: bool = True  # turn off when video window not required
top_down: bool = True  # turn on when working on topdown view
show_depth_frame: bool = True
use_mv: bool = True
print_count = True
bottom = False
second_way: bool = True
wait = 0
cam_dir = os.path.join("camera_data_480p.json")
cam_mat, dist_coeff, _ = load_camera_data_json(cam_dir)
path = "depth_test2"
tello_angles_low = np.loadtxt(os.path.join(path, "tello_angles_low.csv"))
tello_angles_high = np.loadtxt(os.path.join(path, "tello_angles_high.csv"))
cap1 = cv2.VideoCapture(os.path.join(path, "rot_low.h264"))

def overlay_arrows_combined_frames(combined_frame, vectors, max_vectors: int = -1, random_colors: bool = True, vertical=False):
    if max_vectors != -1 and len(vectors) >= max_vectors:
        vectors = vectors[::(vectors.shape[0]//max_vectors)]
    offset = np.array((0, 0, combined_frame.shape[1]//2, 0)) if not vertical else np.array((0, 0, 0, combined_frame.shape[0]//2))
    if random_colors:
        for vector in vectors:
            cv2.line(combined_frame, vector[:2], vector[2:]+offset[2:], np.random.random(size=3)*256)
    else:
        lines = [(vector+offset).reshape((2, 2)) for vector in vectors]  # vectors format to polylines format
        cv2.polylines(combined_frame, lines, isClosed=False, color=(255, 0, 0), thickness=2)
    return combined_frame


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
def generate_angle_pairs(angles1: np.ndarray, angles2: np.ndarray):
    """
    Finds the closest angle from angles2 for each angle in angles1
    """
    distances = np.abs((angles1 % 360)[:, None] - (angles2 % 360))
    distances[distances > 180] = 360 - distances[distances > 180]
    return np.argmin(distances, axis=1)

def depth_from_h264_vectors(vectors: np.ndarray, camera_matrix: np.ndarray, diff: float):
    """
    Estimates depth of each 3d point, represented as a vector(x1, y1, x2, y2) between its
    2d location in 2 consecutive frames with a known height difference
    """
    # some 0s are expected, as detected non-moving points,
    # with non 0 camera movement, the expected distance really is inf.
    assert diff != 0
    if len(vectors) == 0:
        return np.empty(0)
    with np.errstate(divide='ignore'):
        return diff * camera_matrix[1, 1] / np.abs(vectors[:, 3] - vectors[:, 1])

def load_frames(cap):
    frames = []
    for _ in tqdm(repeat(1)):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            return frames


frames_low = load_frames(cap1)
vectors_low = [pi_vectors_conversion(mv) for mv in
               np.load(os.path.join(path, "motion_data_low.npy")).reshape((-1, 30, 41))]
# angles1 = np.loadtxt(os.path.join(path, "tello_angles1.csv"))
# angles2 = np.loadtxt(os.path.join(path, "tello_angles2.csv"))
cap2 = cv2.VideoCapture(os.path.join(path, "rot_high.h264"))
frames_high = load_frames(cap2)
vectors_high = [pi_vectors_conversion(mv) for mv in
                np.load(os.path.join(path, "motion_data_high.npy")).reshape((-1, 30, 41))]
if save_video:
    writer = cv2.VideoWriter(os.path.join(path, "depth.mp4"), -1, 40, (640, 480))
else:
    writer = None  # just to stop warning
# Initialize the feature detector (e.g., ORB, SIFT, etc.)
detector = cv2.ORB_create()
# alignment = generate_angle_pairs(angles_low, angles_high)
alignment = generate_angle_pairs(tello_angles_high, tello_angles_low)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
# for i, (frame_low, angle) in enumerate(zip(frames_low, angles_low)):
frame1 = frames_low[alignment[204]]
frame2 = frames_high[204]
# frame2 = cv2.medianBlur(frame2, 9)
# cap2.set(cv2.CAP_PROP_POS_FRAMES, pair - 1)  # seek to best pair, doesn't work
# combined_frame = np.concatenate((frame1, frame2), axis=1)
# cv2.imshow("frames", combined_frame)
# Motion Vectors
frame1_filter = frame1.copy()
frame2_filter = frame2.copy()

vectors = np.load('vectors.npy')
big_enough = np.abs(vectors[:, 1] - vectors[:, 3]) > 1
vectors = vectors[big_enough]
gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

if second_way:
    kernel1 = np.array([[0, -1, -1],
                        [1, 0, -1],
                        [1, 1, 0]])
    kernel2 = np.array([[-1, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 1]])
    kernel3 = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]])
    kernel4 = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])
    # kernel_tmp = np.array([[1, 1, 1],
    #                    [1, 1, 1],
    #                    [1, 1, 1]])

    output1 = cv2.filter2D(gray, -1, kernel1)
    output2 = cv2.filter2D(gray, -1, kernel2)
    output3 = cv2.filter2D(gray, -1, kernel3)
    output4 = cv2.filter2D(gray, -1, kernel4)

    vectors_filter = np.array([vec for vec in vectors if
                               (output1[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
                                np.mean(output1[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)]) > 2)
                               or (output2[int(vec[3] - 8):int(vec[3] + 8),
                                   int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
                                   np.mean(
                                       output2[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)]) > 2)
                               or (output3[int(vec[3] - 8):int(vec[3] + 8),
                                   int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
                                   np.mean(
                                       output3[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)]) > 2)
                               or (output4[int(vec[3] - 8):int(vec[3] + 8),
                                   int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
                                   np.mean(
                                       output4[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)]) > 2)])


    # vectors_filter = np.array([vec for vec in vectors if
    #                            (output1[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)].size > 0 and
    #                            np.max(output1[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)]) > 24)
    #                            or (output2[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)].size > 0 and
    #                            np.max(output2[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)]) > 24)
    #                            or (output3[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)].size > 0 and
    #                            np.max(output3[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)]) > 24)
    #                            or (output4[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)].size > 0 and
    #                            np.max(output4[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)]) > 24)
    #                             ])

    # vectors_filter = np.array([vec for vec in vectors if
    #                            (output1[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)].size > 0 and
    #                            np.percentile(output1[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)], 50) > 2)
    #                            or (output2[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)].size > 0 and
    #                            np.percentile(output2[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)], 50) > 2)
    #                            or (output3[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)].size > 0 and
    #                            np.percentile(output3[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)], 50) > 2)
    #                            or (output4[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)].size > 0 and
    #                            np.percentile(output4[int(vec[0] - 8):int(vec[0] + 8), int(vec[1] - 8):int(vec[1] + 8)], 50) > 2)
    #                            ])

    def filter_vecs(predicate, kernel, value):
        output = cv2.filter2D(gray, -1, kernel)
        return np.array([vec for vec in vectors if predicate(output[int(vec[0] - 8):int(vec[0] + 8),
                                                             int(vec[1] - 8):int(vec[1] + 8)], 50) > value])
    # vectors_filter = filter_vecs(np.max, kernel_tmp, 24)
else:
    # frame1_filter = cv2.medianBlur(frame1_filter, 9)
    # frame2_filter = cv2.medianBlur(frame2_filter, 9)
    kernel_new = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
    frame1_filter = cv2.filter2D(frame1_filter, -1, kernel_new)
    frame2_filter = cv2.filter2D(frame2_filter, -1, kernel_new)

#     vectors_filter = ffmpeg_encode_extract(frame1_filter, frame2_filter, subpixel=True)
if print_count:
    print(len(vectors), "matches using MV")
if print_count:
    print(len(vectors_filter), "matches using MV after filter")
int_vecs = vectors.astype(int)
int_vecs_filter = vectors_filter.astype(int)

combined_frame = np.concatenate((frame1, frame2), axis=1)
combined_frame_filter = np.concatenate((frame1_filter, frame2_filter), axis=1)

combined_frame = overlay_arrows_combined_frames(combined_frame, int_vecs, max_vectors=50)
combined_frame_filter = overlay_arrows_combined_frames(combined_frame_filter, int_vecs_filter, max_vectors=50)

cv2.imshow("MV matches", combined_frame)
cv2.imshow("MV matches filter", combined_frame_filter)
depth = depth_from_h264_vectors(vectors, cam_mat, 30)
depth_filter = depth_from_h264_vectors(vectors_filter, cam_mat, 30)

points1 = vectors[:, 2:]
points1_vector = vectors_filter[:, 2:]

int_points1 = int_vecs[:, 2:]
int_points1_filter = int_vecs_filter[:, 2:]
if show_depth_frame:
    def show_depth(image1, image2, points_depth, depth_show, name):
        depth_frame = (image1 if bottom else image2).copy()
        depth_color = np.clip(depth_show * 255 / 1000, 0, 255)[:,
                      None]  # clip  values from 0 to 10m and scale to 0-255(color range)
        for color, point in zip(depth_color, points_depth):
            cv2.rectangle(depth_frame, point - 5, point + 5, color, -1)
        if show_video:
            cv2.imshow(f"depth {name}{'MV' if use_mv else 'ORB'}", depth_frame)


    show_depth(frame1, frame2, int_points1, depth, "without filter")
    show_depth(frame1_filter, frame2_filter, int_points1_filter, depth_filter, "with filter")
cv2.waitKey(wait)  # need some minimum time because opencv doesnt work without it