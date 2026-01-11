import cv2
import numpy as np
from sklearn.model_selection import train_test_split

WIDTH = 256
HEIGHT = 256


def _compute_rotation_deltas(neighbor_distance: int) -> np.ndarray:
    angles_deg = np.array(
        [
            0,
            22.5,
            45,
            67.5,
            90,
            112.5,
            135,
            157.5,
            180,
            202.5,
            225,
            247.5,
            270,
            292.5,
            315,
            337.5,
        ],
        dtype=np.float64,
    )
    angles = angles_deg * np.pi / 180.0
    sin_a = np.sin(angles)
    cos_a = np.cos(angles)

    d = np.array([neighbor_distance, 0.0], dtype=np.float64)
    delta = np.zeros((16, 2), dtype=np.float64)

    for i in range(16):
        R = np.array([[cos_a[i], -sin_a[i]], [sin_a[i], cos_a[i]]], dtype=np.float64)
        delta[i] = R @ d  # [dx, dy]

    return delta


def _non_valley_suppression(
    binary_img: np.ndarray, candidates: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    rows, cols = binary_img.shape[:2]
    kept = []

    for c in candidates:
        # neighbors in (x,y)
        neigh = np.round(c.astype(np.float64) + delta).astype(np.int64)

        # keep only in-bound neighbors
        mask = (
            (neigh[:, 0] >= 0)
            & (neigh[:, 0] < cols)
            & (neigh[:, 1] >= 0)
            & (neigh[:, 1] < rows)
        )
        neigh = neigh[mask]
        if neigh.size == 0:
            continue

        values = binary_img[neigh[:, 1], neigh[:, 0]]
        count_non_hand = int(np.sum(values == 0))

        if count_non_hand <= 7:
            kept.append(c.astype(np.int64))

    if len(kept) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    return np.stack(kept, axis=0)


def _threshold_hand_region(
    gray: np.ndarray, blur_sigma: float = 1.0, otsu_threshold: int = 0
) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), blur_sigma)

    # Log transform (avoid division by zero)
    maxv = float(np.max(blurred))
    c = 255.0 / np.log(maxv + 1.0) if maxv > 0 else 1.0
    enhanced = (c * np.log(blurred.astype(np.float64) + 1.0)).astype(np.uint8)

    _, thresholded = cv2.threshold(
        enhanced, otsu_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded


def _largest_contour(binary: np.ndarray) -> np.ndarray:
    # OpenCV version compatibility
    found = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]

    if not contours:
        raise ValueError("No contours found in thresholded image.")

    areas = [cv2.contourArea(c) for c in contours]
    contour = contours[int(np.argmax(areas))]
    return contour


def _contour_center(contour: np.ndarray) -> np.ndarray:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        raise ValueError("Contour moment m00 is zero; cannot compute center.")
    cx = int(m["m10"] // m["m00"])
    cy = int(m["m01"] // m["m00"])
    return np.array([cx, cy], dtype=np.int64)


def _lowpass_distance_signal(
    contour_xy: np.ndarray, center_xy: np.ndarray, freq_threshold: int = 10
) -> np.ndarray:
    dist = np.sqrt(
        np.sum(
            (contour_xy.astype(np.float64) - center_xy.astype(np.float64)) ** 2, axis=1
        )
    )
    freq = np.fft.rfft(dist)
    if freq_threshold < 1:
        raise ValueError("freq_threshold must be >= 1.")
    freq_filtered = np.concatenate([freq[:freq_threshold], 0 * freq[freq_threshold:]])
    dist_smooth = np.fft.irfft(freq_filtered, n=dist.shape[0])
    return dist_smooth


def _find_valley_candidates(
    contour_xy: np.ndarray, smooth_dist: np.ndarray
) -> np.ndarray:
    deriv = np.diff(smooth_dist)
    zero_cross = np.diff(np.sign(deriv)) / 2.0
    idx = np.where(zero_cross > 0)[0]
    if idx.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return contour_xy[idx].astype(np.int64)


def _select_two_valleys_leftmost_top_bottom(valleys: np.ndarray) -> np.ndarray:
    if valleys.shape[0] < 3:
        raise ValueError(
            "Not enough valley candidates (need at least 3 after suppression)."
        )

    order_x = np.argsort(valleys[:, 0])
    triad = valleys[order_x][:3]

    order_y = np.argsort(triad[:, 1])
    chosen = triad[[order_y[0], order_y[2]]]
    return chosen.astype(np.int64)


def _rotate_image_and_points(
    gray: np.ndarray, center: np.ndarray, p0: np.ndarray, p1: np.ndarray
):
    vec = (p1 - p0).astype(np.float64)
    phi = -90.0 + np.arctan2(vec[1], vec[0]) * 180.0 / np.pi

    R = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), phi, 1.0)
    rotated = cv2.warpAffine(gray, R, gray.shape[::-1])

    rp0 = (R[:, :2] @ p0.astype(np.float64) + R[:, 2]).astype(np.int64)
    rp1 = (R[:, :2] @ p1.astype(np.float64) + R[:, 2]).astype(np.int64)

    return rotated, R, rp0, rp1


def _compute_roi_rect(valley0: np.ndarray, valley1: np.ndarray):
    dy = int((valley1[1] - valley0[1]))
    rect0 = (int(valley0[0] + 2 * dy // 6), int(valley0[1] - dy // 6))
    rect1 = (int(valley1[0] + 10 * dy // 6), int(valley1[1] + dy // 6))
    return rect0, rect1


def _crop_with_clamp(img: np.ndarray, rect0, rect1) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = rect0
    x1, y1 = rect1

    # Ensure proper ordering
    x_min, x_max = sorted([x0, x1])
    y_min, y_max = sorted([y0, y1])

    # Clamp
    x_min = max(0, min(w, x_min))
    x_max = max(0, min(w, x_max))
    y_min = max(0, min(h, y_min))
    y_max = max(0, min(h, y_max))

    if x_max - x_min <= 0 or y_max - y_min <= 0:
        raise ValueError("Computed ROI rectangle is invalid after clamping.")

    return img[y_min:y_max, x_min:x_max]


def get_roi(
    image_path: str,
    blur_sigma: float = 1.0,
    otsu_threshold: int = 0,
    freq_threshold: int = 10,
    neighbor_distance: int = 50,
) -> np.ndarray:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    thresholded = _threshold_hand_region(
        gray, blur_sigma=blur_sigma, otsu_threshold=otsu_threshold
    )
    contour = _largest_contour(thresholded)
    center = _contour_center(contour)

    contour_xy = contour[:, 0, :]
    smooth_dist = _lowpass_distance_signal(
        contour_xy, center, freq_threshold=freq_threshold
    )
    candidates = _find_valley_candidates(contour_xy, smooth_dist)

    if candidates.shape[0] == 0:
        raise ValueError("No valley candidates found (zero-crossing minima).")

    delta = _compute_rotation_deltas(neighbor_distance=neighbor_distance)
    suppressed = _non_valley_suppression(thresholded, candidates, delta)

    valleys = _select_two_valleys_leftmost_top_bottom(suppressed)
    valley0, valley1 = valleys[0], valleys[1]

    rotated, _, r0, r1 = _rotate_image_and_points(gray, center, valley0, valley1)
    rect0, rect1 = _compute_roi_rect(r0, r1)
    roi = _crop_with_clamp(rotated, rect0, rect1)

    roi = cv2.resize(roi, (HEIGHT, WIDTH))

    return roi


def m_pq(f, p, q):
    m = 0
    # Loop in f(x,y)
    for x in range(0, len(f)):
        for y in range(0, len(f[0])):
            # +1 is used because if it wasn't, the first row and column would
            # be ignored
            m += ((x + 1) ** p) * ((y + 1) ** q) * f[x][y]
    return m


def centroid(f):
    m_00 = m_pq(f, 0, 0)
    return [m_pq(f, 1, 0) / m_00, m_pq(f, 0, 1) / m_00]


def u_pq(f, p, q):
    u = 0
    centre = centroid(f)
    for x in range(0, len(f)):
        for y in range(0, len(f[0])):
            u += ((x - centre[0] + 1) ** p) * ((y - centre[1] + 1) ** q) * f[x][y]
    return u


def hu(f):
    u_00 = u_pq(f, 0, 0)

    # Scale invariance is obtained by normalization.
    # The normalized central moment is given below
    eta = lambda f, p, q: u_pq(f, p, q) / (u_00 ** ((p + q + 2) / 2))

    # normalized central moments used to compute Hu's seven moments invariat
    eta_20 = eta(f, 2, 0)
    eta_02 = eta(f, 0, 2)
    eta_11 = eta(f, 1, 1)
    eta_12 = eta(f, 1, 2)
    eta_21 = eta(f, 2, 1)
    eta_30 = eta(f, 3, 0)
    eta_03 = eta(f, 0, 3)

    # Hu's moments are computed below
    phi_1 = eta_20 + eta_02
    phi_2 = 4 * eta_11 + (eta_20 - eta_02) ** 2
    phi_3 = (eta_30 - 3 * eta_12) ** 2 + (3 * eta_21 - eta_03) ** 2
    phi_4 = (eta_30 + eta_12) ** 2 + (eta_21 + eta_03) ** 2
    phi_5 = (eta_30 - 3 * eta_12) * (eta_30 + eta_12) * (
        (eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2
    ) + (3 * eta_21 - eta_03) * (eta_21 + eta_03) * (
        3 * (eta_30 + eta_12) - (eta_21 + eta_03) ** 2
    )
    phi_6 = (eta_20 - eta_02) * (
        (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2
    ) + 4 * eta_11 * (eta_30 + eta_12) * (eta_21 + eta_03)
    phi_7 = (3 * eta_21 - eta_03) * (eta_30 + eta_12) * (
        (eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2
    ) - (eta_30 - 3 * eta_12) * (eta_21 + eta_03) * (
        3 * (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2
    )

    return [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]


def process_img(img_path):
    roi = get_roi(img_path)
    features = np.asarray(hu(roi.tolist()))

    return features


def split_train_test(features, labels, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels
    )

    return X_train, X_test, Y_train, Y_test


def knn1_predict(X_train, y_train, x):
    d = np.linalg.norm(X_train - x[None, :], axis=1)
    return int(y_train[np.argmin(d)])


def evaluate_on_test(X_train, X_test, Y_train, Y_test):
    correct = 0
    for i in range(len(X_test)):
        pred = knn1_predict(X_train, Y_train, X_test[i, :])
        correct += int(pred == Y_test[i])

    acc = correct / len(X_test) * 100
    print(f"Accuracy on test: {acc} %")
