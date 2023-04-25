import math
import cv2
import matplotlib.pyplot as plt
import numpy as np

x_c: int
y_c: int


def Dhane_distortion(x_p, y_p, k=1.25, f=1.0):
    """ Takes normalized coordinates and outputs distorted normalized coordinates
    """
    R = np.sqrt(x_p ** 2 + y_p ** 2)
    r = f * np.tan(np.arcsin(np.sin(np.arctan(R / f)) * (1.0 / k)))
    reduction_factor = r / R
    x_dist = reduction_factor * x_p
    y_dist = reduction_factor * y_p

    return x_dist, y_dist


def fisheye_distortion(x_p, y_p, k1, k2):
    """ Takes normalized coordinates and outputs distorted coordinates. This is the fisheye function
        as used by openCV: https://docs.opencv.org/trunk/db/d58/group__calib3d__fisheye.html.
        In principle uninvertible, although a lookup-table can be easily made (see plot_distortion_functions).
    """
    r = np.sqrt(x_p ** 2 + y_p ** 2);
    theta = np.arctan(r);
    # theta_d = theta * (1 + k[0] * theta ** 2 + k[1] * theta ** 4 + k[2] * theta ** 6 + k[3] * theta ** 8);
    theta_d = theta * (1 + k1 * theta ** 2 + k2 * theta ** 4);
    x_dist = (theta_d / r) * x_p;
    y_dist = (theta_d / r) * y_p;

    return x_dist, y_dist;


def apply_distortion(img: np.ndarray, K: np.ndarray, k1: float, k2: float = 0) -> np.ndarray:
    # distort_img = img.copy()
    # v = np.zeros([3, 1])
    # min_y = -2
    # max_y = 2
    # step_y = 0.01
    # min_x = -2
    # max_x = 2
    # step_x = 0.01
    #
    # x = np.arange(min_x, max_x, step_x)
    # y = np.arange(min_y, max_y, step_y)
    # N = len(x)
    # Undistorted = np.zeros([N, N, 3], dtype=np.uint8)

    distort_img = np.zeros(img.shape)
    for x_c in range(img.shape[0] - 1):
        for y_c in range(img.shape[1] - 1):
    # for i in range(N):
    #     for j in range(N):
    #         x_c = x[i]
    #         y_c = y[j]

            # print(f'x = {x_c}, y = {y_c}')
            X = int(np.round(x_c * (1 + k1 * (x_c ^ 2 + y_c ^ 2) + k2 * (x_c ^ 2 + y_c ^ 2))))
            Y = int(np.round(y_c * (1 + k1 * (x_c ^ 2 + y_c ^ 2) + k2 * (x_c ^ 2 + y_c ^ 2))))
            print(X, Y)
            # X_, Y_ = Dhane_distortion(x_c, y_c, k1)
            # print(X_)
            # X_f, Y_f = fisheye_distortion(x_c, y_c, k1, k2)
            # if not np.isnan(X_):
            #     v[0] = X_
            #     v[1] = Y_
            #     v[2] = 1
            #     hom_coord_dist = np.dot(K, v)
            #     x_rounded = int(np.round(hom_coord_dist[0]))
            #     y_rounded = int(np.round(hom_coord_dist[1]))
            #     # print(x_rounded, y_rounded)
            if img.shape[0] > X >= 0 and 0 <= Y < img.shape[1]:

                    # Undistorted[j, i, :] = np.mean(img[y_rounded, x_rounded, :])
                    # print(x_rounded, y_rounded)
                    # print(x_c, y_c)

                distort_img[x_c, y_c, :] = img[X, Y, :]

    return distort_img


plt.imsave()
path_img = 'data/chess.jpg'

img = cv2.imread(path_img)
K = np.asarray([[311.59304538, 0., 158.37457814],
                [0., 313.01338397, 326.49375925],
                [0., 0., 1.]])

d_img = apply_distortion(img, K, 0.9, 0)

cv2.imwrite('chess_dist.jpg', d_img*255)

# def undistortion(img: np.ndarray, k1: float, k2: float = 0) -> np.ndarray:
#     undist_img = img.copy()
#     for x_c in range(img.shape[0]):
#         for y_c in range(img.shape[1]):
#             undist_img = x_c * (1 + k1 * (x_c ^ 2 + y_c ^ 2) + k2 * (x_c ^ 2 + y_c ^ 2))
#
#     return undist_img
