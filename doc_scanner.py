# -*- coding: utf-8 -*-
# @Author: RangeKing
# @Original Author: yilin(https://github.com/insaneyilin/document_scanner)
# Python version: 3.8

import os
import sys
import math
import argparse
import numpy as np
import cv2


def find_corners_by_approx_contour(input_image):
    corners = []
    image = input_image.copy()
    # convert to grayscale and detect edges
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged_image = cv2.Canny(gray_image, 50, 100)
    cv2.imshow("edged", edged_image)
    cv2.waitKey(0)
    # find contours
    cntrs, _ = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = sorted(cntrs, key = cv2.contourArea, reverse=True)[:5]
    # loop over the contours, find approx with 4 points
    for c in cntrs:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            corners = approx
            break

    return corners


def find_corners_by_hough_line_detect(input_image):
    corners = []
    return corners


def get_document_corners(input_image):
    corners = find_corners_by_approx_contour(input_image)
    return [[pt[0][0], pt[0][1]] for pt in corners]


def get_mass_center(points):
    x, y = 0, 0
    for pt in points:
        x += pt[0]
        y += pt[1]
    x /= float(len(points))
    y /= float(len(points))
    return x, y


def sort_rect_points(points):
    mass_center = get_mass_center(points)
    top_pts = []
    bottom_pts = []
    for pt in points:
        if pt[1] < mass_center[1]:
            top_pts.append(pt)
        else:
            bottom_pts.append(pt)

    if len(top_pts) > 2:
        idx = np.argmax(top_pts, axis=0)[1]
        bottom_pts.append(top_pts[idx])
        top_pts.pop(idx)
    if len(bottom_pts) > 2:
        idx = np.argmin(bottom_pts, axis=0)[1]
        top_pts.append(bottom_pts[idx])
        bottom_pts.pop(idx)

    tl = top_pts[0] if top_pts[0][0] < top_pts[1][0] else top_pts[1]
    tr = top_pts[1] if top_pts[0][0] < top_pts[1][0] else top_pts[0]
    bl = bottom_pts[0] if bottom_pts[0][0] < bottom_pts[1][0] else bottom_pts[1]
    br = bottom_pts[1] if bottom_pts[0][0] < bottom_pts[1][0] else bottom_pts[0]

    return tl, tr, br, bl


def apply_four_point_perspective_transform(input_image, points):
    (tl, tr, br, bl) = sort_rect_points(points)

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left coordinates
    width_1 = math.hypot(br[0]-bl[0], br[1]-bl[1])
    width_2 = math.hypot(tr[0]-tl[0], tr[1]-tl[1])
    max_width = max(int(width_1), int(width_2))

    # compute the height of the new image, which will be the
    # maximum distance between top-right and bottom-right
    # y coordinates or the top-left and bottom-left y coordinates
    height_1 = math.hypot(tr[0]-br[0], tr[1]-br[1])
    height_2 = math.hypot(tl[0]-bl[0], tl[1]-bl[1])
    max_height = max(int(height_1), int(height_2))

    # now that we have dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, bottom-left order
    dst = np.array([
                   [0, 0],
                   [max_width-1, 0],
                   [max_width-1, max_height-1],
                   [0, max_height-1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    rect_pts = np.array([
            [tl[0], tl[1]],
            [tr[0], tr[1]],
            [br[0], br[1]],
            [bl[0], bl[1]]], dtype="float32")
    persp_trans_mat = cv2.getPerspectiveTransform(rect_pts, dst)
    warped_image = cv2.warpPerspective(input_image, persp_trans_mat, (max_width,max_height))
    # return the warped image
    return warped_image


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input_image', type=str, help='input image filename',
            default=None)
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    image = cv2.imread(args.input_image)
    corners = np.array(get_document_corners(image))
    cv2.imshow("input_image", image)
    if len(corners) > 3:
        contour_image = image.copy()
        cv2.drawContours(contour_image, [corners], -1, (0, 255, 0), 2)
        cv2.imshow("contour", contour_image)
    if len(corners) == 4:
        warped_image = apply_four_point_perspective_transform(image, corners)
        cv2.imshow("warped_image", warped_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

