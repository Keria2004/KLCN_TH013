import numpy as np
import imageio
import cv2


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def _imread(image_name):
    # đọc ảnh bằng imageio (mặc định RGB)
    return imageio.imread(image_name)


def _imresize(image_array, size):
    # cv2.resize nhận (width, height), trong khi size thường là (h, w)
    return cv2.resize(image_array, (size[1], size[0]))


def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical
