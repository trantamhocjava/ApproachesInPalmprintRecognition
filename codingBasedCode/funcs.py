import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from functools import reduce
from scipy.signal import convolve2d
from sklearn.decomposition import KernelPCA

HEIGHT = 256
WIDTH = 256


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

    contour_xy = contour[:, 0, :]  # (N,2) [x,y]
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


def _normalize_to_0_255(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.min(x)
    denom = np.max(x)
    if denom < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x / denom) * 255.0


def derivate_image(im: np.ndarray, angle: str) -> np.ndarray:
    h, w = im.shape
    pad_im = np.pad(im, ((1, 0), (1, 0)), mode="edge")
    # pad_im shape: (h+1, w+1)

    if angle == "horizontal":
        # d/dx (left difference)
        # align to im: take pad_im[1: , :w] minus im
        return pad_im[1:, :w] - im
    elif angle == "vertical":
        # d/dy (up difference)
        return pad_im[:h, 1:] - im
    else:
        raise ValueError("angle must be 'horizontal' or 'vertical'")


def extract_ltrp1(im_d_x: np.ndarray, im_d_y: np.ndarray) -> np.ndarray:
    encoded = np.zeros_like(im_d_y, dtype=np.uint8)

    encoded[np.logical_and(im_d_x >= 0, im_d_y >= 0)] = 1
    encoded[np.logical_and(im_d_x < 0, im_d_y >= 0)] = 2
    encoded[np.logical_and(im_d_x < 0, im_d_y < 0)] = 3
    encoded[np.logical_and(im_d_x >= 0, im_d_y < 0)] = 4
    return encoded


def extract_ltrp2(ltrp1_code: np.ndarray) -> np.ndarray:
    h, w = ltrp1_code.shape
    padded = np.pad(ltrp1_code, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    g_c1 = np.zeros((3, h, w), dtype=np.uint8)
    g_c2 = np.zeros((3, h, w), dtype=np.uint8)
    g_c3 = np.zeros((3, h, w), dtype=np.uint8)
    g_c4 = np.zeros((3, h, w), dtype=np.uint8)

    # Neighborhood (clockwise) around (i,j) in padded coords
    # Fixed bug: j-1 (NOT j-11)
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            g_c = padded[i, j]

            nb = np.array(
                [
                    padded[i + 1, j],  # down
                    padded[i + 1, j - 1],  # down-left
                    padded[i, j - 1],  # left
                    padded[i - 1, j - 1],  # up-left
                    padded[i - 1, j],  # up
                    padded[i - 1, j + 1],  # up-right
                    padded[i, j + 1],  # right
                    padded[i + 1, j + 1],  # down-right
                ],
                dtype=np.uint8,
            )

            mask = nb != g_c
            ltrp2_local = nb * mask  # zeros where same as g_c

            def _bin8(bits: np.ndarray) -> int:
                # bits is bool[8] -> int [0..255]
                # keep same reduce logic but safe for python3
                return int(
                    reduce(lambda a, b: (a << 1) | int(b), bits.astype(np.uint8), 0)
                )

            if g_c == 1:
                for idx, direction in enumerate((2, 3, 4)):
                    g_dir = ltrp2_local == direction
                    g_c1[idx, i - 1, j - 1] = _bin8(g_dir)
            elif g_c == 2:
                for idx, direction in enumerate((1, 3, 4)):
                    g_dir = ltrp2_local == direction
                    g_c2[idx, i - 1, j - 1] = _bin8(g_dir)
            elif g_c == 3:
                for idx, direction in enumerate((1, 2, 4)):
                    g_dir = ltrp2_local == direction
                    g_c3[idx, i - 1, j - 1] = _bin8(g_dir)
            elif g_c == 4:
                for idx, direction in enumerate((1, 2, 3)):
                    g_dir = ltrp2_local == direction
                    g_c4[idx, i - 1, j - 1] = _bin8(g_dir)
            else:
                # padded zeros can occur; just skip (leave zeros)
                pass

    large = np.concatenate([g_c1, g_c2, g_c3, g_c4], axis=0)  # (12,h,w)
    return large


def extract_ltrp2_hist(
    ltrp2_code: np.ndarray, block_size: int = 8, no_bins: int = 8, hist_range=(0, 256)
) -> np.ndarray:
    P, H, W = ltrp2_code.shape
    if H % block_size != 0 or W % block_size != 0:
        raise ValueError("Image side must be divisible by block_size")

    n_blocks_h = H // block_size
    n_blocks_w = W // block_size

    features_per_p = []
    for p_idx in range(P):
        comp = ltrp2_code[p_idx].astype(np.uint8)

        rows = []
        for bi in range(n_blocks_h):
            row_hist = []
            i0 = bi * block_size
            i1 = i0 + block_size
            for bj in range(n_blocks_w):
                j0 = bj * block_size
                j1 = j0 + block_size
                block = comp[i0:i1, j0:j1]

                # histogram with fixed bins
                hist = cv2.calcHist(
                    [block],
                    channels=[0],
                    mask=None,
                    histSize=[no_bins],
                    ranges=list(hist_range),
                )
                hist = hist.flatten().astype(np.float32)
                s = hist.sum()
                if s > 1e-8:
                    hist /= s
                row_hist.append(hist)

            rows.append(np.concatenate(row_hist, axis=0))

        features_per_p.append(
            np.stack(rows, axis=0)
        )  # (n_blocks_h, n_blocks_w*no_bins)

    # Stack P-components vertically
    return np.concatenate(features_per_p, axis=0)  # (P*n_blocks_h, n_blocks_w*no_bins)


def extract_compcode_with_magnitude(
    input_image: np.ndarray, no_theta: int = 12, sigma: float = 1.5
):
    theta = np.arange(1, no_theta + 1) * np.pi / no_theta

    # filter size 35x35 like original snippet
    x, y = np.meshgrid(np.arange(0, 35, 1), np.arange(0, 35, 1))
    xo = x.shape[0] / 2.0
    yo = y.shape[0] / 2.0

    # same equations as your code
    kappa = np.sqrt(2.0 * np.log(2.0)) * ((2.0**sigma + 1.0) / (2.0**sigma - 1.0))
    omega = kappa / sigma

    responses = []
    for t in theta:
        xp = (x - xo) * np.cos(t) + (y - yo) * np.sin(t)
        yp = -(x - xo) * np.sin(t) + (y - yo) * np.cos(t)

        psi = (
            (-omega / (np.sqrt(2 * np.pi)) * kappa)
            * np.exp((-(omega**2) / (8 * (kappa**2))) * (4 * (xp**2) + (yp**2)))
            * (np.cos(omega * xp) - np.exp(-(kappa**2) / 2))
        )

        filtered = convolve2d(input_image, psi, mode="same", boundary="symm")
        responses.append(filtered)

    responses = np.stack(responses, axis=0)  # (no_theta,H,W)
    orientations = np.argmin(responses, axis=0).astype(np.uint8)
    magnitude = np.min(responses, axis=0).astype(np.float32)

    return orientations, magnitude


def derivate_image_palm_line(
    im: np.ndarray, angle: str, m1: int = 3, m2: int = 1
) -> np.ndarray:
    pad = np.pad(im, ((m2, m1), (m2, m1)), mode="edge")
    H, W = im.shape
    out = np.zeros_like(im, dtype=np.float32)

    if angle == "horizontal":
        for i in range(m2, m2 + H):
            for j in range(m2, m2 + W):
                g_c = pad[i, j]

                e1_sum = 0.0
                for k in range(0, m1):
                    e1_sum += pad[i, j + k]
                element1 = (e1_sum + g_c) / float(m1 + 1)

                e2_sum = 0.0
                for k in range(0, m2):
                    e2_sum += (
                        pad[i, j + k] + pad[i - k, j] + pad[i + k, j] + pad[i, j + k]
                    )
                element2 = e2_sum / float(m2 * 4)

                out[i - m2, j - m2] = element1 - element2

    elif angle == "vertical":
        for i in range(m2, m2 + H):
            for j in range(m2, m2 + W):
                g_c = pad[i, j]

                e1_sum = 0.0
                for k in range(0, m1):
                    e1_sum += pad[i + k, j]
                element1 = (e1_sum + g_c) / float(m1 + 1)

                e2_sum = 0.0
                for k in range(0, m2):
                    e2_sum += (
                        pad[i, j + k] + pad[i - k, j] + pad[i + k, j] + pad[i, j + k]
                    )
                element2 = e2_sum / float(m2 * 4)

                out[i - m2, j - m2] = element1 - element2
    else:
        raise ValueError("angle must be 'horizontal' or 'vertical'")

    return out


def extract_local_tetra_pattern_palm(
    image: np.ndarray,
    input_mode: str = "gabor",  # 'grayscale' or 'gabor'
    theta_orientations: int = 12,
    comp_sigma: float = 1.5,
    derivative_mode: str = "palmprint",  # 'standard' or 'palmprint'
    m1: int = 3,
    m2: int = 1,
    block_size: int = 8,
    n_bins: int = 8,
    pca_no_components: int = 15,
) -> np.ndarray:
    img = image.astype(np.float32)

    if input_mode == "gabor":
        _, mag = extract_compcode_with_magnitude(
            img, no_theta=theta_orientations, sigma=comp_sigma
        )
        # convert min-response to 0..255 as in your notebook (invert then normalize)
        mag = (mag - np.max(mag)) * -1.0
        img = _normalize_to_0_255(mag)
    elif input_mode == "grayscale":
        img = _normalize_to_0_255(img)
    else:
        raise ValueError("input_mode must be 'grayscale' or 'gabor'")

    if derivative_mode == "standard":
        dx = derivate_image(img, "horizontal")
        dy = derivate_image(img, "vertical")
    elif derivative_mode == "palmprint":
        dx = derivate_image_palm_line(img, "horizontal", m1=m1, m2=m2)
        dy = derivate_image_palm_line(img, "vertical", m1=m1, m2=m2)
    else:
        raise ValueError("derivative_mode must be 'standard' or 'palmprint'")

    ltrp1 = extract_ltrp1(dx, dy)
    ltrp2 = extract_ltrp2(ltrp1)
    ltrp2_hist = extract_ltrp2_hist(
        ltrp2, block_size=block_size, no_bins=n_bins, hist_range=(0, 256)
    )

    # KernelPCA (linear kernel) like your notebook
    kpca = KernelPCA(n_components=pca_no_components, kernel="linear")
    feat_2d = kpca.fit_transform(ltrp2_hist.astype(np.float32))
    return feat_2d.astype(np.float32)


def process_image(img_path) -> np.ndarray:
    roi = get_roi(img_path)
    feat_2d = extract_local_tetra_pattern_palm(roi)

    # Flatten to 1D vector
    return feat_2d.flatten()


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
