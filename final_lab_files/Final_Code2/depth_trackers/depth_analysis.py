import cv2
import numpy as np
from scipy.spatial.transform import Rotation


#  can be a lot faster
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


def cloud_from_vectors(vectors: np.ndarray, camera_matrix: np.ndarray, height_diff: float,
                       frame: np.ndarray = None) -> np.ndarray:
    """
    create a cloud from h264 vectors
    Parameters:
        vectors: (n, 4) array of vectors between 2 frames, returns point cloud relative to the second frame.
        camera_matrix: camera intrinsic matrix.
        height_diff: height difference between the frames.
        frame: frame to reference colors from(the second frame)
    Returns:
        a (n,3) cloud generated from the vectors, or a (n,6) colored cloud if a frame was given as well.
    """
    assert height_diff > 0
    if len(vectors) == 0:
        return np.empty(0)
    twodpoints = vectors[:, 2:]
    if frame is not None:  # use colored point cloud
        pixels = twodpoints.astype(int)
        cloud = np.empty((vectors.shape[0], 6))
        cloud[:, 3:] = frame[pixels[:, 1], pixels[:, 0]] / 255  # select pixels to copy colors from
    else:
        cloud = np.empty((vectors.shape[0], 3))
    # negative because image and world coordinates are backwards
    cloud[:, :2] = np.squeeze(-cv2.undistortPoints(twodpoints[None, :, :].astype(np.float32, copy=False), camera_matrix, None))
    # some 0s are expected, as detected non-moving points,
    # with non 0 camera movement, the expected distance really is inf.
    with np.errstate(divide='ignore'):
        cloud[:, 2] = height_diff * camera_matrix[1, 1] / np.abs(vectors[:, 3] - vectors[:, 1])
    cloud = cloud[np.isfinite(cloud[:, 2])]
    cloud[:, :2] *= cloud[:, 2:3]
    return cloud


def cloud_to_topdown(cloud: np.ndarray, resolution: int = 300, maximum_distance: float = 1000) -> np.ndarray:
    """
    Turns a 3d cloud(colored or not) to a topdown view frame
    Parameters:
        cloud: (n,3) uncolored cloud or (n,6) colored cloud
        resolution: the resolution to use for the frame
        (the generated result will be of shape (resolution, resolution, 3))
        maximum_distance: the maximum distance(in any single direction, not a radius) to display in the result,
         further away points will be clipped to the edge.
    Returns:
        (resolution,resolution) image(3 layer array) of a topdown view of the given cloud
    """
    image = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
    points = np.clip(cloud[:, (0, 2)]/maximum_distance*resolution+resolution/2, 0, resolution-1).astype(int)
    for i, point in enumerate(points):
        # colored cloud or not
        color = (0, 0, 0) if cloud.shape[1] == 3 else (cloud[i, 3:] * 255).astype(np.uint8)
        image[point[1], point[0]] = color
    return image


def depth_from_pi_vectors(vectors: np.ndarray, camera_matrix: np.ndarray, diff, **kwargs):
    """
    Estimates depth of each 3d point, represented in
     a picamera motion vectors matrix(MxN matrix with (x_diff, y_diff, SAD) in each cell)
     between 2 consecutive frames with a known height difference
    """
    assert diff != 0
    with np.errstate(divide='ignore'):
        return diff * camera_matrix[1, 2] / np.abs(vectors["y"])

# def triangulate_from_vectors(vectors, loc1, loc2, rot1, rot2, cam_mat, distortion_coeff = None):
    # rot1_mat = Rotation.from_euler('y', rot1)
    # rot2_mat = Rotation.from_euler('y', rot1)
    # projMat1 = np.hstack(rot1, loc1))  # origin
    # projMat2 = np.hstack(rot1., loc2))  # R, T
    # # projMat2 = np.hstack([np.eye(3), np.array((0, diff, 0))[:, None]])  # R, T
    # points1 = vectors[:, :2]
    # points2 = vectors[:, 2:]
    # points1u = cv2.undistortPoints(points1, cam_mat, distortion_coeff)
    # points2u = cv2.undistortPoints(points2, cam_mat, distortion_coeff)
    # points4d = cv2.triangulatePoints(projMat1, projMat2, points1u, points2u)
    # points3d = (points4d[:3, :] / points4d[3, :]).T

def triangulate_points(keypoints1, keypoints2, matches, diff, cam_mat, distortion_coeff):
    projMat1 = np.hstack([np.eye(3), np.zeros((3, 1))])  # origin
    projMat2 = np.hstack([np.eye(3), np.array((0, diff, 0))[:, None]])  # R, T
    points1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
    points2 = np.array([keypoints2[match.trainIdx].pt for match in matches])
    points1u = cv2.undistortPoints(points1, cam_mat, distortion_coeff)
    points2u = cv2.undistortPoints(points2, cam_mat, distortion_coeff)
    points4d = cv2.triangulatePoints(projMat1, projMat2, points1u, points2u)
    points3d = (points4d[:3, :] / points4d[3, :]).T
    return points3d


if __name__ == "__main__":
    import os.path
    from mapping.utils.Constants import Constants

    path = os.path.join(Constants.ROOT_DIR, "results/depth_test1")
    angles1 = np.loadtxt(os.path.join(path, "tello_angles1.csv"))
    angles2 = np.loadtxt(os.path.join(path, "tello_angles2.csv"))
    best_pair = generate_angle_pairs(angles1, angles2)
    cap1 = cv2.VideoCapture(os.path.join(path, "rot1.h264"))
    cap2 = cv2.VideoCapture(os.path.join(path, "rot2.h264"))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, best_pair[0] - 1)  # seek to best pair
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    combined_frame = np.concatenate((frame1, frame2), axis=1)
    cv2.imshow("frames", combined_frame)
    cv2.waitKey(0)
