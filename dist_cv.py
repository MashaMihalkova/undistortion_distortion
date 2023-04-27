# Add radial distortion to GS image
from typing import List, Tuple, Optional

import numpy as np
import cv2
import math
import statistics

from warpOF import gen_flow_shift, image_warp

FX = 1000 // 8
FY = 1000 // 8


def apply_dist(img: np.ndarray, coefficients: List[float], warp_coef: Optional[float]=None) -> np.ndarray:
    rows, cols, chn = img.shape

    dst_img = np.zeros((rows, cols, chn))

    # Set internal parameters
    fx = FX  # 2414.55  # 779.423
    fy = FY  # 2413.54  # 779.423
    cx = img.shape[0] // 2
    cy = img.shape[1] // 2

    # Radial distortion parameters
    k1 = coefficients[0]  # 0.274058
    k2 = coefficients[1]  # 0  #-1.56158
    k3 = coefficients[2]  # 0  #2.86023
    mismatch_x = []
    mismatch_y = []
    for j in range(rows):
        for i in range(cols):
            # Go to camera coordinate system
            x = (i - cx) / fx
            y = (j - cy) / fy
            r = x * x + y * y

            # Add radial distortion
            x_un = x * (1 + k1 * r + k2 * r * r + k3 * r * r * r)
            y_un = y * (1 + k1 * r + k2 * r * r + k3 * r * r * r)

            newx, newy = calc_undistortion_coeff(x, y)

            # if round(newx, 4) != round(x_un, 4):
            #     mismatch_x.append(round(newx, 4) - round(x_un, 4))
            #     print(f'newx = {round(newx, 4)}, x = {round(x, 4)}')
            # if round(newy, 4) != round(y_un, 4):
            #     mismatch_y.append(round(newy, 4) - round(y_un, 4))
            #     print(f'newy = {round(newy, 4)}, y = {round(y, 4)}')

            # Then go to the image coordinate system
            if warp_coef is not None:
                u = newx * fx + cx + warp_coef
                v = newy * fy + cy + warp_coef
            else:

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

    print(f'len = {len(mismatch_x)}')
    print(f'len = {len(mismatch_y)}')
    if mismatch_x:
        print(f'max x = {max(mismatch_x)}')
        print(f'min x = {min(mismatch_x)}')
        print(f'avg x = {statistics.mean(map(abs, mismatch_x))}')
    if mismatch_y:
        print(f'max y = {max(mismatch_y)}')
        print(f'min y = {min(mismatch_y)}')
        print(f'avg y = {statistics.mean(map(abs, mismatch_y))}')

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


# def apply_undist(img: np.ndarray, coefficients: List[float]) -> Tuple[np.ndarray, np.ndarray]:
def apply_undist(img: np.ndarray, coefficients: List[float], warp_coef: Optional[float] = None) -> np.ndarray:
    rows, cols, chn = img.shape

    dst_img = np.zeros((rows, cols, chn))
    dst_img_ = np.zeros((cols, rows, chn))
    # Set internal parameters
    fx = FX
    fy = FY
    cx = img.shape[0] // 2
    cy = img.shape[1] // 2
    mismatch_x = []
    mismatch_y = []
    for y_j in range(rows):
        for x_i in range(cols):

            # Go to camera coordinate system
            x = (y_j - cx) / fx
            y = (x_i - cy) / fy

            #
            if x_i != 0 and y_j != 0:

                # und_x, und_y = calc_undistortion_coeff(x, y)
                r = x * x + y * y
                newx = x * (1 + coefficients[0] * r)
                newy = y * (1 + coefficients[0] * r)

                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # region check x==und_x and y==und_y
                # r = und_x * und_x + und_y * und_y
                # newx = und_x * (1 + coefficients[0] * r)
                # newy = und_y * (1 + coefficients[0] * r)
                und_x = newx
                und_y = newy

                # if round(newx, 4) != round(x, 4):
                #     mismatch_x.append(round(newx, 4) - round(x, 4))
                #     # print(f'newx = {round(newx, 4)}, x = {round(x, 4)}')
                # if round(newy, 4) != round(y, 4):
                #     mismatch_y.append(round(newy, 4) - round(y, 4))
                #     # print(f'newy = {round(newy, 4)}, y = {round(y, 4)}')
                # endregion
            else:
                r = x * x + y * y
                newx = x * (1 + coefficients[0] * r)
                newy = y * (1 + coefficients[0] * r)
                und_x = newx
                und_y = newy

                # und_x = x_i
                # und_y = y_j

            # Then go to the image coordinate system
            if warp_coef is not None:
                u = und_x * fx + cx + warp_coef
                v = und_y * fy + cy + warp_coef
            else:
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

    print(f'len = {len(mismatch_x)}')
    print(f'len = {len(mismatch_y)}')
    if mismatch_x:
        print(f'max x = {max(mismatch_x)}')
        print(f'min x = {min(mismatch_x)}')
        print(f'avg x = {statistics.mean(map(abs, mismatch_x))}')
    if mismatch_y:
        print(f'max y = {max(mismatch_y)}')
        print(f'min y = {min(mismatch_y)}')
        print(f'avg y = {statistics.mean(map(abs, mismatch_y))}')

    # return dst_img, dst_img_
    return dst_img_


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def main_method(path_img: str, const_shift: int) -> None:
    # region apply distortion  (S2)
    img = cv2.imread(path_img, 1)
    dst_chess2 = apply_dist(img, [0.2, 0, 0])
    cv2.imwrite("data_test/S2.jpg", dst_chess2)
    # endregion

    # region undistort obtained dist_img  (R2)
    _, undist_chess2 = apply_undist(dst_chess2, [0.2, 0, 0])
    cv2.imwrite("data_test/R2.jpg", undist_chess2)
    # endregion

    # region apply warp OF  (R1)
    const_shift = const_shift
    flow = gen_flow_shift(height=undist_chess2.shape[0], width=undist_chess2.shape[1], shift=const_shift)
    warp_chess1_undist = image_warp(undist_chess2, flow)
    cv2.imwrite('data_test/R1.png', warp_chess1_undist)
    # endregion

    # region apply distortion to warp_chess1  (S1)
    dst_chess1 = apply_dist(warp_chess1_undist, [0.2, 0, 0])
    cv2.imwrite("data_test/S1.jpg", dst_chess1)
    # endregion


def developed_method(path_img: str, const_shift: float) -> None:
    R2 = cv2.imread(path_img, 1)
    new_S1 = np.zeros_like(R2)
    new_S1_w = np.zeros_like(R2)
    flow = gen_flow_shift(height=R2.shape[0], width=R2.shape[1], shift=const_shift)


    # new_S1 = image_warp(apply_undist(R2, [0.2, 0, 0]), flow)
    # new_S1_w = apply_undist(R2, [0.2, 0, 0], const_shift)
    # new_S1 = image_warp(apply_dist(R2, [0.2, 0, 0]), flow)
    new_S1 = image_warp(apply_dist(R2, [0.2, 0, 0]), flow)

    # new_S1_minus_w = apply_dist(R2, [0.2, 0, 0], -0.5)
    new_S1_plus_w = apply_dist(R2, [0.2, 0, 0], 0.5)

    cv2.imwrite('dataset/dis_S1.png', new_S1)
    # cv2.imwrite('dataset/dis_S1_minus_w.png', new_S1_minus_w)
    cv2.imwrite('dataset/dis_S1_plus_w.png', new_S1_plus_w)


# main_method(path_img="data/chess3.jpg", const_shift=4)
# S1 = cv2.imread("dataset/big_img/R1.png", 1)
# S2 = cv2.imread("dataset/big_img/R2.jpg", 1)
# scale_percent = 8 # percent of original size
# width = int(S1.shape[1] / scale_percent)
# height = int(S1.shape[0] / scale_percent )
# S1_r = cv2.resize(S1, (width, height))
# S2_r = cv2.resize(S2, (width, height))
# cv2.imwrite('dataset/res_R1.png', S1_r)
# cv2.imwrite('dataset/res_R2.png', S2_r)
#
# S1 = cv2.imread("dataset/big_img/S1.jpg", 1)
# S2 = cv2.imread("dataset/big_img/S2.jpg", 1)
# scale_percent = 8 # percent of original size
# width = int(S1.shape[1] / scale_percent)
# height = int(S1.shape[0] / scale_percent )
# S1_r = cv2.resize(S1, (width, height))
# S2_r = cv2.resize(S2, (width, height))
# cv2.imwrite('dataset/res_S1.png', S1_r)
# cv2.imwrite('dataset/res_S2.png', S2_r)

# R1_res = cv2.imread("dataset/res_R1.png", 1)
# R2_res = cv2.imread("dataset/res_R2.png", 1)
#
# S1 = apply_dist(R1_res, [0.2, 0, 0])
# cv2.imwrite("dataset/S1_res.png", S1)
# S2 = apply_dist(R2_res, [0.2, 0, 0])
# cv2.imwrite("dataset/S2_res.png", S2)


developed_method(path_img="dataset/res_R2.png", const_shift=0.5)


# fx = 1000  # 2414.55  # 779.423
# fy = 1000
# cx = dst_img.shape[0] // 2
# cy = dst_img.shape[1] // 2
# camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# und_cv = cv2.undistort(dst_img, camera_matrix, np.array((-0.2, 0, 0, 0)))
# cv2.imwrite("chess_undist_openCV.jpg", und_cv)
