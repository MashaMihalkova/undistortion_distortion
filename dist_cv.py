# Add radial distortion to GS image
from typing import List, Tuple

import numpy as np
import cv2
import math


def apply_dist(img: np.ndarray, coefficients: List[float]) -> np.ndarray:
    rows, cols, chn = img.shape

    dst_img = np.zeros((rows, cols, chn))

    # Set internal parameters
    fx = 779.423  # 2414.55  # 779.423
    fy = 779.423  # 2413.54  # 779.423
    cx = img.shape[0] // 2
    cy = img.shape[1] // 2

    # Radial distortion parameters
    k1 = coefficients[0]  # 0.274058
    k2 = coefficients[1]  # 0  #-1.56158
    k3 = coefficients[2]  # 0  #2.86023

    for j in range(rows):
        for i in range(cols):
            # Go to camera coordinate system
            x = (i - cx) / fx
            y = (j - cy) / fy
            r = x * x + y * y

            # Add radial distortion
            newx = x * (1 + k1 * r + k2 * r * r + k3 * r * r * r)
            newy = y * (1 + k1 * r + k2 * r * r + k3 * r * r * r)

            # Then go to the image coordinate system
            u = newx * fx + cx
            v = newy * fy + cy

            # Bilinear interpolation
            u0 = math.floor(u)  # //Equivalent to the x coordinate in the above formula is 0
            v0 = math.floor(v)  # //Equivalent to the y ordinate in the above formula is 0
            u1 = u0 + 1  # //Equivalent to the x coordinate in the above formula is 1
            v1 = v0 + 1  # //Equivalent to the y ordinate in the above formula is 1

            dx = u - u0  # //Here dx is equivalent to x in the above formula
            dy = v - v0  # //Here dy is equivalent to y in the above formula
            weight1 = (1 - dx) * (1 - dy)
            weight2 = dx * (1 - dy)
            weight3 = (1 - dx) * dy
            weight4 = dx * dy

            if u0 >= 0 and u1 < cols and v0 >= 0 and v1 < rows:
                dst_img[j, i, :] = weight1 * img[v0, u0, :] + weight3 * img[v1, u0, :] + \
                                   weight2 * img[v0, u1, :] + weight4 * img[v1, u1, :]

    return dst_img


def calc_undistortion_coeff(i: int, j: int) -> Tuple[float, float]:
    X = i
    Y = j
    r = (X ** 2 + Y ** 2)
    if X != 0 and Y != 0 and r != 0:
        sigma_f = (5 * Y ** 3) / (2 * r)
        sigma_s = math.sqrt(((25 * Y ** 6) / (4 * r ** 2)) + ((125 * Y ** 6) / (27 * r ** 3)))
        sigma2 = (sigma_f + sigma_s) ** (1 / 3)
        if sigma2 != 0:
            sigma1 = sigma2 - ((5 * Y ** 2) / (3 * r * sigma2))

            und_x = X * sigma1 / Y
            und_y = sigma1
        else:
            und_x = X
            und_y = Y
    else:
        und_x = X
        und_y = Y
    return und_x, und_y


def apply_undist(img: np.ndarray, coefficients: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols, chn = img.shape

    dst_img = np.zeros((rows, cols, chn))
    dst_img_ = np.zeros((cols, rows, chn))
    # Set internal parameters
    fx = 1000
    fy = 1000
    cx = img.shape[0] // 2
    cy = img.shape[1] // 2

    for y_j in range(rows):
        for x_i in range(cols):

            # Go to camera coordinate system
            x = (y_j - cx) / fx
            y = (x_i - cy) / fy

            #
            if x_i != 0 and y_j != 0:

                und_x, und_y = calc_undistortion_coeff(x, y)
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # region check x==und_x and y==und_y
                r = und_x * und_x + und_y * und_y
                newx = und_x * (1 + 0.2 * r)
                newy = und_y * (1 + 0.2 * r)
                # und_x = newx
                # und_y = newy

                if round(newx, 4) != round(x, 4):
                    print(f'newx = {round(newx, 4)}, x = {round(x, 4)}')
                if round(newy, 4) != round(y, 4):
                    print(f'newy = {round(newy, 4)}, y = {round(y, 4)}')
                # endregion
            else:

                und_x = x_i
                und_y = y_j

            # Then go to the image coordinate system
            u = und_x * fx + cx
            v = und_y * fy + cy

            # Bilinear interpolation
            u0 = math.floor(u)  # //Equivalent to the x coordinate in the above formula is 0
            v0 = math.floor(v)  # //Equivalent to the y ordinate in the above formula is 0
            u1 = u0 + 1  # //Equivalent to the x coordinate in the above formula is 1
            v1 = v0 + 1  # //Equivalent to the y ordinate in the above formula is 1

            dx = u - u0  # //Here dx is equivalent to x in the above formula
            dy = v - v0  # //Here dy is equivalent to y in the above formula
            weight1 = (1 - dx) * (1 - dy)
            weight2 = dx * (1 - dy)
            weight3 = (1 - dx) * dy
            weight4 = dx * dy

            if u0 >= 0 and u1 < cols and v0 >= 0 and v1 < rows:
                # dst_img[y_j, x_i, :] = weight1 * img[v0, u0, :] + weight3 * img[v1, u0, :] + \
                #                        weight2 * img[v0, u1, :] + weight4 * img[v1, u1, :]

                dst_img[v0, u0, :] = weight1 * img[x_i, y_j, :] + weight3 * img[x_i, y_j, :] + \
                                     weight2 * img[x_i, y_j, :] + weight4 * img[x_i, y_j, :]

                dst_img_[x_i, y_j, :] = weight1 * img[v0, u0, :] + weight3 * img[v1, u0, :] + \
                                        weight2 * img[v0, u1, :] + weight4 * img[v1, u1, :]

    return dst_img, dst_img_


#
# def apply_undist_(img: np.ndarray, coefficients: List[float]) -> np.ndarray:
#     rows, cols, chn = img.shape
#     fx = 1000 # 779.423  # 2414.55  # 779.423
#     fy = 1000 # 779.423  # 2413.54  # 779.423
#     cx = img.shape[0] // 2
#     cy = img.shape[1] // 2
#     for j in range(rows):
#         for i in range(cols):
#             newx, newy = undistort(i, j, coefficients, 3)
#             # Then go to the image coordinate system
#             # u = newx * fx + cx
#             # v = newy * fy + cy
#             #
#             # # Bilinear interpolation
#             # u0 = math.floor(u)  # //Equivalent to the x coordinate in the above formula is 0
#             # v0 = math.floor(v)  # //Equivalent to the y ordinate in the above formula is 0
#             # u1 = u0 + 1  # //Equivalent to the x coordinate in the above formula is 1
#             # v1 = v0 + 1  # //Equivalent to the y ordinate in the above formula is 1
#             #
#             # dx = u - u0  # //Here dx is equivalent to x in the above formula
#             # dy = v - v0  # //Here dy is equivalent to y in the above formula
#             # weight1 = (1 - dx) * (1 - dy)
#             # weight2 = dx * (1 - dy)
#             # weight3 = (1 - dx) * dy
#             # weight4 = dx * dy
#             #
#             # if u0 >= 0 and u1 < cols and v0 >= 0 and v1 < rows:
#             dst_img[j, i, :] = img[int(newx), int(newy), :]
#             # (1 - dx) * (1 - dy) * img[v0, u0, :] + (1 - dx) * dy * img[v1, u0, :] + dx * (
#             #             1 - dy) * img[v0, u1, :] + dx * dy * img[v1, u1, :]
#
#     return dst_img

#
# def undistort(x, y, coefficients, iter_num=3):
#     rows, cols, chn = img.shape
#
#     dst_img = np.zeros((rows, cols, chn))
#
#     # Set internal parameters
#     fx = 779.423  # 2414.55  # 779.423
#     fy = 779.423  # 2413.54  # 779.423
#     cx = img.shape[0] // 2
#     cy = img.shape[1] // 2
#
#     # Radial distortion parameters
#     k1 = coefficients[0]
#     k2 = coefficients[1]
#     k3 = coefficients[2]
#     p1 = 0
#     p2 = 0
#     # k1, k2, p1, p2, k3 = distortion
#     # fx, fy = k[0, 0], k[1, 1]
#     # cx, cy = k[:2, 2]
#     x = x
#     y = y
#     x = (x - cx) / fx
#     x0 = x
#     y = (y - cy) / fy
#     y0 = y
#     for _ in range(iter_num):
#         r2 = x ** 2 + y ** 2
#         k_inv = 1 / (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
#         delta_x = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
#         delta_y = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
#         x = (x0 - delta_x) * k_inv
#         y = (y0 - delta_y) * k_inv
#     return np.array((x * fx + cx, y * fy + cy))


img = cv2.imread("data/chess.jpg", 1)
# img = cv2.resize(img, (388,388))
# cv2.imwrite('cat_res.jpg', img)
# dst_img = apply_dist(img, [0.2, 0, 0])
# cv2.imwrite("chess_dist_1000.jpg", dst_img)

dst_img = cv2.imread("chess_dist_1000.jpg", 1)
img_und, img_und_ = apply_undist(dst_img, [0.2, 0, 0])
cv2.imwrite("chess_undist_1000.jpg", img_und)
cv2.imwrite("chess_undist_1000_.jpg", img_und_)

# fx = 1000  # 2414.55  # 779.423
# fy = 1000
# cx = dst_img.shape[0] // 2
# cy = dst_img.shape[1] // 2
# camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# und_cv = cv2.undistort(dst_img, camera_matrix, np.array((-0.2, 0, 0, 0)))
# cv2.imwrite("chess_undist_openCV.jpg", und_cv)
