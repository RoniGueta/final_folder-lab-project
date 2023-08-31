from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from analysis.MotionArrowsOverlay import overlay_arrows_combined_frames
from depth_trackers.depth_analysis import cloud_from_vectors


class Mapper:
    def __init__(self, cam_mat: np.ndarray, vector_generator, twod_filter=(lambda vectors, frame: vectors),
                 threed_filter=(lambda cloud: cloud), final_threed_filter=(lambda cloud: cloud),
                 estimate_angle: bool = False):
        """
        Parameters:
            cam_mat: camera intrinsics matrix
            vector_generator: function that given 2 frames, generates vectors between them,
            must return array of shape (n,4) where each row is (x1,y1,x2,y2)
            twod_filter: function that given vectors and a frame, filters the vectors using the frame
            threed_filter: function that given a cloud, filters it.
            estimate_angle: whether to estimate angles based on the frames or use a given angle
        """
        self.twod_filter = twod_filter
        self.threed_filter = threed_filter
        self.final_threed_filter = final_threed_filter
        self.reference_frames = list()
        self.reference_locations = list()
        self.reference_angles_queue = list()
        self.reference_angles = np.empty(0)
        self.estimate_angle = estimate_angle
        self.vector_generator = vector_generator
        self.cam_mat = cam_mat
        self.__cloud_queue = deque([np.empty((0, 3))])  # init with empty cloud
        self.target_angles = set()
        if estimate_angle:
            from angle_trackers.angle_estimators import angle_diff_split_tan
            from angle_trackers.point_pairs_generators.motion_vectors import MotionVectors
            from angle_trackers.vector_tracker import VectorTracker
            self.angle_estimator_reference = VectorTracker(MotionVectors, angle_diff_split_tan, cam_mat,
                                                           cumulative=True, median=True)
            self.angle_estimator_target = VectorTracker(MotionVectors, angle_diff_split_tan, cam_mat, cumulative=True,
                                                        median=True)

    def register_reference_frame(self, frame: np.ndarray, rotation: float = None, location=(0, 100, 0),
                                 motion_vectors: np.ndarray = None, deduplicate_angles: bool = False):
        """
        Register a frame into the reference pool of frames
        Parameters:
            frame: the frame to register
            rotation: the yaw of the frame
            location: the location of the camera
            motion_vectors: motion vectors between this frame and the previous reference frame
            (Required only when estimate_rotation is set to True)
            deduplicate_angles: whether to check if the existing angle is already registered, and if it is,
             do not register thw new frame
        """
        assert rotation is not None or (self.estimate_angle and motion_vectors is not None)
        if deduplicate_angles and rotation in self.reference_angles_queue or rotation in self.reference_angles:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        self.reference_frames.append(frame)
        if self.estimate_angle:
            rotation = self.angle_estimator_reference.track_frame(motion_vectors)
        self.reference_angles_queue.append(rotation % 360)
        self.reference_locations.append(location)

    def __prepare_data(self):
        """
        Organizes data to make cloud generation faster
        """
        self.reference_angles = np.concatenate((self.reference_angles, self.reference_angles_queue))
        self.reference_angles_queue = list()

    def __find_reference_frame(self, rotation: float) -> (np.ndarray, (float, float, float)):
        """
        find the frame at the closest rotation to the given one
        Parameters:
            rotation(float): the yaw to search
        Returns:
            (np.ndarray, (float, float, float): the frame and its location
        """
        distances = np.abs(self.reference_angles - (rotation % 360))
        distances[distances > 180] = 360 - distances[distances > 180]
        index = np.argmin(distances)
        return self.reference_frames[index], self.reference_locations[index]

    def register_target_frame(self, frame: np.ndarray, rotation: float = None, location=(0, 130, 0),
                              motion_vectors: np.ndarray = None, annotate_frame: bool = False, show_matches: bool = False, direct: bool = False, deduplicate_angles: bool = True):
        """
        Registers a target frame by finding a close rotation match within the reference frames,
        creating a point cloud from their vectors and adding it to the global cloud.
        Parameters:
            frame(np.ndarray): the target frame
            rotation(float): the yaw of the frame
            location(float, float, float): the location of the camera.
            motion_vectors(np.ndarray): motion vectors between the target frame and the previous target frame.
            (Required only when estimate_angles is set to True)
            annotate_frame: whether to draw depths on the given target frame
        """
        assert rotation is not None or (self.estimate_angle and motion_vectors is not None)
        if deduplicate_angles:
            if rotation in self.target_angles:
                return
            self.target_angles.add(rotation)
        if len(self.reference_angles_queue) != 0:  # move angles from queue to array
            self.__prepare_data()
        if self.estimate_angle:
            rotation = self.angle_estimator_target.track_frame(motion_vectors)
        reference_frame, reference_location = self.__find_reference_frame(rotation)
        converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        vectors = self.vector_generator(reference_frame, converted_frame)
        if len(vectors) == 0:
            return
        if direct:
            new_cloud = vectors
        else:
            vectors = self.twod_filter(vectors=vectors, frame=frame)
            if show_matches:
                combined_frame = np.concatenate((cv2.cvtColor(reference_frame, cv2.COLOR_YUV2BGR_I420), frame), axis=1)
                combined_frame = overlay_arrows_combined_frames(combined_frame, vectors.astype(int), max_vectors=50, vertical=False)
                cv2.imshow("matches", combined_frame)
            new_cloud = cloud_from_vectors(vectors, camera_matrix=self.cam_mat,
                                           height_diff=abs(location[1] - reference_location[1]), frame=None)
        new_cloud[:, :3] += location
        rotation = Rotation.from_euler('y', rotation, degrees=True)  # change to 3 direction rotation when possible
        new_cloud = self.threed_filter(new_cloud)
        if annotate_frame and len(new_cloud) > 1:
            # image and world coordinates are backwards
            image_coords = new_cloud[None, :, :3].copy()
            image_coords[0, :, :2] *= -1
            twodpoints = np.squeeze(cv2.projectPoints(image_coords, (0, 0, 0),
                                                      location, self.cam_mat,
                                                      None)[0]).astype(int)
            colors = np.clip(new_cloud[:, 2]/1000*255, 0, 255).astype(np.uint8)
            for point, color in zip(twodpoints, colors):
                cv2.rectangle(frame, (point - 5),(point + 5), int(color), -1)
        # move rotation before annotate when figure out good way to calc depth after rotation
        new_cloud[:, :3] = new_cloud[:, :3] @ rotation.as_matrix()
        self.__add_cloud(new_cloud)

    def __add_cloud(self, cloud):
        self.__cloud_queue.append(cloud)

    @property
    def cloud(self):
        """
        The cloud generated so far
        """
        if len(self.__cloud_queue) > 1:
            self.cloud = np.concatenate(self.__cloud_queue)
        return self.__cloud_queue[0]

    @cloud.setter
    def cloud(self, cloud):
        self.__cloud_queue = deque([cloud])

    def get_open3d_cloud(self, clean=True):
        import open3d as o3d
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self.finalize_cloud() if clean else self.cloud[:, :3])
        filtered_cloud, _ = cloud.remove_statistical_outlier(3, 0.7)
        return filtered_cloud

    def finalize_cloud(self):
        return self.final_threed_filter(self.cloud)