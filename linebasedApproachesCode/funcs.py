import os
from enum import Enum
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Whole hand contour extraction
THRESHOLD_VALUE = 40
BORDER_THICKNESS = 10
WRIST_PORTION = 1 / 6

# ROI parameters
WIDTH_MODIFIER = 1.2
ROI_START_OFFSET = 30

# Feature extraction parameters
CONTOUR_COUNT = 3


def get_contours(original_image):
    # Remove right 1/6 of the image since it contains only wrist and distracting features
    height, width, channels = original_image.shape
    borderless_image = original_image[:, : int(width * (1 - WRIST_PORTION))]
    height, width, channels = borderless_image.shape

    borderless_image_original = borderless_image.copy()

    # Binary thresholding for distinguishing palm from the background
    _, borderless_image = cv2.threshold(
        borderless_image, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY
    )

    # problem: sometimes the tips of the fingers are exceeding the confines of the image
    # solution: set a few bottom, top and right pixel rows to black, so the contours will continue to the wrist
    # Create a new image with the black border
    image = np.zeros(
        (height + 2 * BORDER_THICKNESS, width + 2 * BORDER_THICKNESS, 3), dtype=np.uint8
    )
    image[
        BORDER_THICKNESS : BORDER_THICKNESS + height,
        BORDER_THICKNESS : BORDER_THICKNESS + width,
    ] = borderless_image
    height, width, channels = image.shape

    # Apply dilation and Gaussian blur to remove jagged edges and help connect the contours into one
    kernel = np.ones((9, 9), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.GaussianBlur(image, (9, 9), 0)

    # Detect hand contour
    canny = cv2.Canny(image, 10, 47)

    # Blur the canny image a bit to fix loosely connected contours
    canny = cv2.GaussianBlur(canny, (3, 3), 0)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def get_longest_contour(contours):
    longest_contour = None
    longest_contour_length = 0

    for contour in contours:
        length = cv2.arcLength(contour, True)

        if length > longest_contour_length:
            longest_contour_length = length
            longest_contour = contour

    return longest_contour


def get_points(longest_contour):
    hull = cv2.convexHull(longest_contour, returnPoints=False)
    hull[::-1].sort(axis=0)

    defects = cv2.convexityDefects(longest_contour, hull)

    points = []

    # Iterate through the defects and find significant ones
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d > 20000:
            points.append(tuple(longest_contour[f][0]))

    # Sort points by y value
    points.sort(key=lambda p: p[1])

    pointA = points[0]  # between ring and little finger
    pointB = points[2]  # between index and middle finger

    return pointA, pointB


def get_roi_params(pointA, pointB):
    midpoint = ((pointA[0] + pointB[0]) // 2, (pointA[1] + pointB[1]) // 2)
    vector = (pointB[0] - pointA[0], pointB[1] - pointA[1])
    perpendicular_vector = (vector[1], -vector[0])

    perpendicular_vector_length = np.linalg.norm(perpendicular_vector)
    vector_length = np.linalg.norm(vector)

    normalized_perpendicular_vector = (
        perpendicular_vector[0] / perpendicular_vector_length,
        perpendicular_vector[1] / perpendicular_vector_length,
    )
    normalized_vector = (vector[0] / vector_length, vector[1] / vector_length)

    roi_start = (
        int(midpoint[0] + normalized_perpendicular_vector[0] * ROI_START_OFFSET),
        int(midpoint[1] + normalized_perpendicular_vector[1] * ROI_START_OFFSET),
    )

    roi_side_length_half = int(
        np.linalg.norm(perpendicular_vector) * WIDTH_MODIFIER / 2
    )

    vertices = np.array(
        [
            [
                int(roi_start[0] + normalized_vector[0] * roi_side_length_half),
                int(roi_start[1] + normalized_vector[1] * roi_side_length_half),
            ],
            [
                int(roi_start[0] - normalized_vector[0] * roi_side_length_half),
                int(roi_start[1] - normalized_vector[1] * roi_side_length_half),
            ],
            [
                int(
                    roi_start[0]
                    - normalized_vector[0] * roi_side_length_half
                    + normalized_perpendicular_vector[0] * roi_side_length_half * 2
                ),
                int(
                    roi_start[1]
                    - normalized_vector[1] * roi_side_length_half
                    + normalized_perpendicular_vector[1] * roi_side_length_half * 2
                ),
            ],
            [
                int(
                    roi_start[0]
                    + normalized_vector[0] * roi_side_length_half
                    + normalized_perpendicular_vector[0] * roi_side_length_half * 2
                ),
                int(
                    roi_start[1]
                    + normalized_vector[1] * roi_side_length_half
                    + normalized_perpendicular_vector[1] * roi_side_length_half * 2
                ),
            ],
        ],
        dtype=np.int32,
    )

    vertices = vertices.reshape((-1, 1, 2))

    return vertices, vector


def find_rotation(vector, image):
    angle_radians = np.arctan2(vector[1], vector[0])
    angle_degrees = np.degrees(angle_radians)

    height, width, channels = image.shape
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees - 180, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image, rotation_matrix


def get_roi(original_image, vertices, rotation_matrix):
    rotated_vertices = cv2.transform(vertices, rotation_matrix)

    rotated_vertices = rotated_vertices.reshape(-1, 2)

    roi = original_image[
        rotated_vertices[0][1] : rotated_vertices[2][1],
        rotated_vertices[0][0] : rotated_vertices[1][0],
    ]
    non_equalized = roi.copy()

    roi = cv2.equalizeHist(roi[:, :, 0])
    roi = np.repeat(roi[:, :, np.newaxis], 3, axis=2)

    return roi, non_equalized


def get_gabor(roi):
    angle_range = np.arange(1.4, 6.3, 0.2)
    images = []

    # Gabor filter parameters
    ksize = (31, 31)  # Kernel size
    sigma = 0.8  # Standard deviation
    lambd = 25  # Wavelength
    psi = 0  # Phase offset

    for theta in angle_range:
        kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, psi)
        roi_blur = cv2.GaussianBlur(roi, (11, 11), 5)
        roi_gray = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.filter2D(roi_gray, -1, kernel)
        images.append(filtered_image)

    dst = np.average(images, axis=0)
    dst = np.uint8(dst)
    dst = cv2.resize(dst, (188 * 2, 187 * 2), interpolation=cv2.INTER_NEAREST)

    return dst


def get_features(roi):
    def get_contours(image):
        clahe = cv2.createCLAHE(clipLimit=1)
        image = clahe.apply(image)

        adaptive_threshold = cv2.adaptiveThreshold(
            -image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -5
        )

        contours, _ = cv2.findContours(
            adaptive_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        sorted_contours = sorted(
            contours, key=lambda x: cv2.arcLength(x, True), reverse=True
        )

        return sorted_contours[:CONTOUR_COUNT]

    gabor = get_gabor(roi)

    gabor = (gabor - gabor.min()) / (gabor.max() - gabor.min()) * 255
    gabor = gabor.astype("uint8")

    contours1 = get_contours(gabor)

    if len(contours1) < CONTOUR_COUNT:
        return False, np.array([[0, 0, 0] for i in range(CONTOUR_COUNT)])

    contours1_moments = [cv2.moments(contour1) for contour1 in contours1]

    contours1_y = [x["m01"] / (x["m00"] + 1e-8) for x in contours1_moments]

    contours1 = sorted(zip(contours1_y, contours1), key=lambda x: x[0])
    _, contours1 = zip(*contours1)

    np.set_printoptions(suppress=True)

    c = []

    for contour in contours1:
        reshaped_array = contour.reshape((len(contour), 2))

        # Split the reshaped array into two separate arrays
        x = reshaped_array[:, 0]
        y = reshaped_array[:, 1]

        coefficients = np.polyfit(x, y, 2)

        c.append(coefficients)

    return True, c


def process_image(img_path):
    original_image = cv2.imread(img_path)

    contours = get_contours(original_image)
    longest_contour = get_longest_contour(contours)

    pointA, pointB = get_points(longest_contour)
    vertices, vector = get_roi_params(pointA, pointB)
    rotated_image, rotation_matrix = find_rotation(vector, original_image)
    roi, non_equalized = get_roi(rotated_image, vertices, rotation_matrix)
    found, features = get_features(roi)

    if found == False:
        raise Exception("Not found features for roi")

    features = np.array(features).reshape(-1)

    return features


def split_train_test(features, labels, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels
    )

    return X_train, X_test, Y_train, Y_test


def knn1_predict(X_train, y_train, x):
    """1-NN với Euclidean distance (template matching kiểu đơn giản)."""
    d = np.linalg.norm(X_train - x[None, :], axis=1)
    return int(y_train[np.argmin(d)])


def evaluate_on_test(X_train, X_test, Y_train, Y_test):
    correct = 0
    for i in range(len(X_test)):
        pred = knn1_predict(X_train, Y_train, X_test[i, :])
        correct += int(pred == Y_test[i])

    acc = correct / len(X_test) * 100
    print(f"Accuracy on test: {acc} %")
