import cv2
import numpy as np


def height_filter(cloud, minimum_height=50, maximum_height=150):
    return cloud[(cloud[:, 1] > minimum_height) & (cloud[:, 1] < maximum_height)]


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


def organize_local(point1, point2, points: np.ndarray, percentile=20):
    if points.shape[0] == 0:
        return None
    new_point = np.zeros(3)
    new_point[1] = (point1[1] + point2[1]) / 2
    new_point[2] = (point1[2] + point2[2]) / 2
    # new_point[0] = np.mean(points[:, 0])
    new_point[0] = np.percentile(points[:, 0], percentile)
    return new_point


def organize_local2d(point1, point2, points: np.ndarray):
    if points.shape[0] == 0:
        return None
    new_point = np.zeros(2)
    new_point[1] = (point1[1] + point2[1]) / 2
    new_point[0] = np.percentile(points[:, 0], 10)
    return new_point


def cart2D2pol2D(cart2D_cloud):
    x_2 = np.power(cart2D_cloud[:, 0], 2)
    y_2 = np.power(cart2D_cloud[:, 1], 2)
    rho = np.sqrt(np.add(x_2, y_2))
    phi = np.arctan2(cart2D_cloud[:, 1], cart2D_cloud[:, 0])
    new_cloud = np.array([rho, phi])
    return new_cloud.T


def pol2D2cart2D(pol2D_cloud):
    x = np.multiply(pol2D_cloud[:, 0], np.cos(pol2D_cloud[:, 1]))
    y = np.multiply(pol2D_cloud[:, 0], np.sin(pol2D_cloud[:, 1]))
    new_cloud = np.array([x, y])
    return new_cloud.T


def cart3D2pol3D(cart3D_cloud):
    x = cart3D_cloud[:, 0]
    y = cart3D_cloud[:, 1]
    z = cart3D_cloud[:, 2]
    x_2 = x ** 2
    y_2 = y ** 2
    z_2 = z ** 2
    xy = np.sqrt(x_2 + y_2)
    pol = np.empty_like(cart3D_cloud)
    pol[:, 0] = np.sqrt(x_2 + y_2 + z_2)
    pol[:, 1] = np.arctan2(y, x)
    pol[:, 2] = np.arctan2(xy, z)
    return pol


def pol3D2cart3D(pol3D_cloud):
    r = pol3D_cloud[:, 0]
    theta = pol3D_cloud[:, 1]
    phi = pol3D_cloud[:, 2]
    cart3D = np.array([r * np.cos(theta) * np.sin(phi),
                       r * np.sin(theta) * np.sin(phi),
                       r * np.cos(phi)
                       ])
    cart3D = np.array(np.vsplit(np.transpose(cart3D), cart3D.shape[1]))
    cart3D = np.array([c[0] for c in cart3D])
    return cart3D


def radius_filter(cloud, jump_size=3, window_size=5, angle_window=(-180, 180), camera_location=(0, 100, 0), percentile=20):
    assert cloud.shape[1] == 3 and jump_size <= window_size
    cloud -= camera_location
    cloud = cart3D2pol3D(cloud)
    cloud[:, [1, 2]] = np.rad2deg(cloud[:, [1, 2]])  # convert radians to degrees
    cloud1 = []
    for i in range(angle_window[0], angle_window[1], jump_size):
        for j in range(0, 180, jump_size):
            # points = np.array([p for p in cloud if i <= p[1] <= i + x_window_size and j <= p[2] <= j + y_window_size])
            filter = (cloud[:, 1] <= i + window_size) & (cloud[:, 1] >= i)
            filter &= (cloud[:, 2] <= j + window_size)
            filter &= (cloud[:, 2] >= j)
            points = cloud[filter]
            ret = organize_local(np.array([0, i, j]), np.array([0, i + window_size, j + window_size]), points, percentile)
            if ret is not None:
                cloud1.append(ret)
    cloud = np.array(cloud1)
    cloud[:, [1, 2]] = np.deg2rad(cloud[:, [1, 2]])  # convert radians to degrees
    cloud = pol3D2cart3D(cloud)
    return cloud


def radius_2d_filter(cloud, jump_size=1, window_size=2, angle_window=(-180, 180), camera_location=(0, 100, 0)):
    assert jump_size <= window_size
    cloud -= camera_location
    cloud = cloud[:, [0, 2]]
    cloud = cart2D2pol2D(cloud)
    cloud[:, 1] = np.rad2deg(cloud[:, 1])  # convert radians to degrees
    cloud1 = []
    for i in range(angle_window[0], angle_window[1], jump_size):
        points = cloud[(cloud[:, 1] <= i + window_size) & (cloud[:, 1] >= i)]
        ret = organize_local2d(np.array([0, i]), np.array([0, i + window_size]), points)
        if ret is not None:
            cloud1.append(ret)
    cloud = np.array(cloud1)
    cloud[:, 1] = np.deg2rad(cloud[:, 1])  # convert radians to degrees
    cloud = pol2D2cart2D(cloud)
    return cloud
