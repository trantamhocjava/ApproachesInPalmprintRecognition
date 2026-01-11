import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from skimage import morphology
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split


def Low_pass_Gausian_process(img, D0):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    Locx = np.array(list(range(cols)) * rows).reshape([rows, cols])
    Locy = np.transpose((np.array(list(range(rows)) * cols).reshape([cols, rows])))
    D = np.sqrt((Locx - cols / 2) ** 2 + (Locy - rows / 2) ** 2)
    mask = np.exp(-(D**2) / D0**2 / 2)

    f1 = np.fft.fft2(img)
    f1shift = np.fft.fftshift(f1)
    f1shift = f1shift * mask
    f2shift = np.fft.ifftshift(f1shift)
    img_new = np.fft.ifft2(f2shift)
    img_new = np.abs(img_new)

    return img_new


def gabor_wavelet(rows, cols, orientation, scale, n_orientation):
    kmax = np.pi / 2
    f = np.sqrt(2)
    delt2 = (2 * np.pi) ** 2
    k = (kmax / (f**scale)) * np.exp(1j * orientation * np.pi / n_orientation / 2)
    kn2 = np.abs(k) ** 2
    gw = np.zeros((rows, cols), np.complex128)

    for m in range(int(-rows / 2) + 1, int(rows / 2) + 1):
        for n in range(int(-cols / 2) + 1, int(cols / 2) + 1):
            t1 = np.exp(-0.5 * kn2 * (m**2 + n**2) / delt2)
            t2 = np.exp(1j * (np.real(k) * m + np.imag(k) * n))
            t3 = np.exp(-0.5 * delt2)
            gw[int(m + rows / 2 - 1), int(n + cols / 2 - 1)] = (
                (kn2 / delt2) * t1 * (t2 - t3)
            )

    return gw


class Gabor:
    def __init__(self, R, C, n_orientation, scale):
        self.R = R
        self.C = C
        self.n_orientarion = n_orientation
        self.scale = scale
        self.orientation = np.array(
            [u * np.pi / n_orientation for u in range(1, n_orientation + 1)]
        )
        self.gabor_filters_sets = [
            gabor_wavelet(R, C, u, scale, n_orientation)
            for u in range(1, n_orientation + 1)
        ]

    def filtering(self, img):
        graphs = np.array(
            [cv2.filter2D(img, -1, np.real(gw)) for gw in self.gabor_filters_sets]
        )
        return graphs

    def plot_filters(self, n_scale):
        gabor_filters = []
        fig = plt.figure()
        for v in range(1, n_scale + 1):
            for u in range(1, self.n_orientarion + 1):
                gw = gabor_wavelet(self.R, self.C, u, v, self.n_orientarion)
                fig.add_subplot(
                    n_scale, self.n_orientarion, self.n_orientarion * (v - 1) + u
                )
                plt.imshow(np.real(gw), cmap="gray")
        plt.show()


n_orientation = 6
scale = 2
GA = Gabor(10, 10, n_orientation, scale)
gabor_filters = GA.gabor_filters_sets


HEIGHT = 128
WIDTH = 128


def LOG_preprocess(img, R0=40, ksize=5):
    AfterGaussian = np.uint8(Low_pass_Gausian_process(img, R0))
    processed = cv2.Laplacian(AfterGaussian, -1, ksize=ksize)
    img = cv2.equalizeHist(img)
    return processed


def process(img, gabor_filters):
    img = LOG_preprocess(img)

    After_gabor = []
    for i, gw in enumerate(gabor_filters):
        element = cv2.filter2D(img, -1, np.real(gw))
        After_gabor.append(element)

    Two_value = []
    for i, line in enumerate(After_gabor):
        _, TW = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)
        TW = cv2.erode(TW, kernel)

        Two_value.append(TW)
    con = []
    for i in Two_value:
        conective = morphology.remove_small_objects(i > 0, min_size=40, connectivity=1)
        con.append(conective)

    line = np.sum(con, axis=0) / len(con)

    return line


def block_entropy(block, eps=1e-12, bins=16):
    b = block.astype(np.float32)
    b = (b - b.min()) / (b.max() - b.min() + eps)
    hist, _ = np.histogram(b.ravel(), bins=bins, range=(0.0, 1.0), density=True)
    hist = hist + eps
    return float(-np.sum(hist * np.log(hist)))


def local_stat_feature(response_map, block=16):
    rm = response_map.astype(np.float32)
    rm = cv2.normalize(rm, None, 0.0, 1.0, cv2.NORM_MINMAX)

    H, W = rm.shape
    H2 = (H // block) * block
    W2 = (W // block) * block
    rm = rm[:H2, :W2]  # cắt cho chia hết block

    feats = []
    for y in range(0, H2, block):
        for x in range(0, W2, block):
            b = rm[y : y + block, x : x + block]
            mu = float(np.mean(b))
            sd = float(np.std(b))
            energy = float(np.mean(b * b))
            ent = block_entropy(b, bins=16)
            feats.extend([mu, sd, energy, ent])
    return np.array(feats, dtype=np.float32)


def get_local_stat_feature_for_1_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (HEIGHT, WIDTH))

    resp = process(img, gabor_filters)
    feat = local_stat_feature(resp)

    return feat


def knn1_predict(X_train, y_train, x):
    d = np.linalg.norm(X_train - x[None, :], axis=1)
    return int(y_train[np.argmin(d)])


def split_train_test(features, labels, test_size):
    X_train, X_test, Y_train, Y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels
    )

    return X_train, X_test, Y_train, Y_test


def evaluate_on_test(X_train, X_test, Y_train, Y_test):
    correct = 0
    for i in range(len(X_test)):
        pred = knn1_predict(X_train, Y_train, X_test[i, :])
        correct += int(pred == Y_test[i])

    acc = correct / len(X_test) * 100
    print(f"Accuracy on test: {acc} %")
