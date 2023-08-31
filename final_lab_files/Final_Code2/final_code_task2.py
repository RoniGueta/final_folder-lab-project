import os
import time
import cv2
import numpy as np
from tqdm import tqdm
from depth_trackers.depth_analysis import cloud_to_topdown
from depth_trackers.depth_tracker_filters import height_filter
from depth_trackers.mapper import Mapper
from file import load_camera_data_json
from mv_extractor.ffmpeg_enc import MVExtractor, ffmpeg_encode_extract

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
def emboss_filter(vectors, frame, threshold=5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output1 = cv2.filter2D(gray, -1, kernel1)
    output2 = cv2.filter2D(gray, -1, kernel2)
    output3 = cv2.filter2D(gray, -1, kernel3)
    output4 = cv2.filter2D(gray, -1, kernel4)
    vectors = vectors.astype(int)
    return np.array([vec for vec in vectors if
                     ((output1[(vec[3] - 8):(vec[3] + 8), (vec[2] - 8):(vec[2] + 8)]).size > 0 and
                      np.mean(output1[(vec[3] - 8):(vec[3] + 8), (vec[2] - 8):(vec[2] + 8)]) > threshold)
                     or ((output2[(vec[3] - 8):(vec[3] + 8), (vec[2] - 8):(vec[2] + 8)]).size > 0 and
                         np.mean(output2[(vec[3] - 8):(vec[3] + 8), (vec[2] - 8):(vec[2] + 8)]) > threshold)
                     or ((output3[(vec[3] - 8):(vec[3] + 8), (vec[2] - 8):(vec[2] + 8)]).size > 0 and
                         np.mean(output3[(vec[3] - 8):(vec[3] + 8), (vec[2] - 8):(vec[2] + 8)]) > threshold)
                     or ((output4[(vec[3] - 8):(vec[3] + 8), (vec[2] - 8):(vec[2] + 8)]).size > 0 and
                         np.mean(output4[(vec[3] - 8):(vec[3] + 8), (vec[2] - 8):(vec[2] + 8)]) > threshold)])

    # return np.array([vec for vec in vectors if
    #                  (output1[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
    #                   np.mean(output1[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)]) > threshold)
    #                  or (output2[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
    #                      np.mean(output2[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)]) > threshold)
    #                  or (output3[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
    #                      np.mean(output3[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)]) > threshold)
    #                  or (output4[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
    #                      np.mean(output4[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)]) > threshold)])
    # return np.array([vec for vec in vectors if
    #                  (output1[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
    #                  np.percentile(output1[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)], 50) > threshold)
    #                  or (output2[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
    #                  np.percentile(output2[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)], 50) > threshold)
    #                  or (output3[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
    #                  np.percentile(output3[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)], 50) > threshold)
    #                  or (output4[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)].size > 0 and
    #                  np.percentile(output4[int(vec[3] - 8):int(vec[3] + 8), int(vec[2] - 8):int(vec[2] + 8)], 50) > threshold)])

cam_dir = os.path.join("camera_data_480p.json")
cam_mat, dist_coeff, _ = load_camera_data_json(cam_dir)
path = "depth_test1"
tello_angles_low = np.loadtxt(os.path.join(path, "tello_angles_low.csv"))
tello_angles_high = np.loadtxt(os.path.join(path, "tello_angles_high.csv"))

save_video: bool = False  # turn off when saved video not required
show_video: bool = False  # turn off when video window not required
top_down: bool = False  # turn on when working on topdown view
show_depth_frame: bool = False
show_matches = False

min_height = 0
max_height = 150
ratio_test = 0.9  # parameter for ratio test
percentile = 30  # parameter for percentile filter
height_diff = 30  # distance between drone locations
mv_extractor = MVExtractor()
writer = cv2.VideoWriter('output_without.h264', cv2.VideoWriter_fourcc(*'h264'), 40, (640, 480))

# mapper = EarlyEncodeMapper(cam_mat=cam_mat, final_threed_filter=lambda cloud: radius_filter(height_filter(cloud, min_height, max_height)), direct=True)
# mapper = Mapper(cam_mat=cam_mat, vector_generator=feature_matching, final_threed_filter=lambda cloud: radius_filter(height_filter(cloud, min_height, max_height)))
# mapper = ParMapper(cam_mat=cam_mat, vector_generator=mv_extractor.extract, twodfilter= emboss_filter, lambda cloud: radius_filter(height_filter(cloud, min_height, max_height)))
mapper = Mapper(cam_mat=cam_mat, vector_generator=ffmpeg_encode_extract, twod_filter=emboss_filter, threed_filter=lambda cloud: height_filter(cloud, min_height, max_height))
cap1 = cv2.VideoCapture(os.path.join(path, "rot_low.h264"))
cap2 = cv2.VideoCapture(os.path.join(path, "rot_high.h264"))
t = time.time()
for angle in tqdm(tello_angles_low):
    ret, frame = cap1.read()
    if not ret:
        break
    mapper.register_reference_frame(frame=frame, rotation=angle, deduplicate_angles=True)
cap1.release()
for i, angle in tqdm(enumerate(tello_angles_high)):
    ret, frame = cap2.read()
    # if i % 3 != 0:
    #     continue
    if not ret:
        break
    mapper.register_target_frame(frame=frame, rotation=angle)
    writer.write(frame)
    # mapper.register_target_frame(frame=frame, rotation=angle, annotate_frame=show_depth_frame, direct=False, deduplicate_angles=True, show_matches=show_matches)
    # if show_depth_frame:
    #     cv2.imshow('depth', frame)
    #     if i % 10 == 0:
    #         cv2.imshow('topdown', cloud_to_topdown(mapper.cloud))
    #     cv2.waitKey(0)
    writer.release()
cv2.imwrite("topdown_without.bmp", cloud_to_topdown(mapper.cloud))
print(time.time()-t)
# np.savetxt('cloud.xyz', np.asarray(mapper.get_open3d_cloud().points))
# o3d.visualization.draw_geometries([mapper.get_open3d_cloud(clean=False)])
# o3d.visualization.draw_geometries([mapper.get_open3d_cloud()])
