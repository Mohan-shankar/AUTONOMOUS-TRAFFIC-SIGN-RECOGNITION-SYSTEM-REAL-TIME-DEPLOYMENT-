"""
Data utilities for AUTONOMOUS TRAFFIC SIGN RECOGNITION SYSTEM
Provides dataset loader, preprocessing, and advanced augmentations:
- Weather (Rain / Fog / Snow)
- Motion Blur
- Occlusion
- Rotation & Scaling
Designed for TensorFlow + EfficientNetB0.
"""

import os
import random
import cv2
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------
# BASIC IMAGE PREPROCESSING
# ---------------------------------------------------------

def read_image(path, target_size=(224, 224)):
    """Reads an image from disk and resizes it."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img


def normalize(img):
    """Normalize image to [0,1]."""
    return img.astype("float32") / 255.0


# ---------------------------------------------------------
# WEATHER AUGMENTATIONS
# ---------------------------------------------------------

def add_rain(img, density=0.0015):
    h, w, _ = img.shape
    img_rain = img.copy()
    num_drops = int(h * w * density)

    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(10, 20)
        x2 = x + np.random.randint(-5, 5)
        y2 = y + length

        cv2.line(img_rain, (x, y), (x2, y2), (200, 200, 200), 1)

    return cv2.blur(img_rain, (3, 3))


def add_fog(img, strength=0.6):
    h, w, _ = img.shape
    fog = np.full((h, w, 3), 255, dtype=np.uint8)
    return cv2.addWeighted(img, 1 - strength, fog, strength, 0)


def add_snow(img, density=0.0012):
    h, w, _ = img.shape
    img_snow = img.copy()
    flakes = int(h * w * density)

    for _ in range(flakes):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        cv2.circle(img_snow, (x, y), 1, (255, 255, 255), -1)

    return cv2.blur(img_snow, (3, 3))


# ---------------------------------------------------------
# MOTION BLUR + CAMERA SHAKE
# ---------------------------------------------------------

def motion_blur(img, degree=10, angle=0):
    kernel = np.zeros((degree, degree))
    kernel[int((degree - 1) / 2), :] = np.ones(degree)

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel = kernel / degree

    return cv2.filter2D(img, -1, kernel)


def camera_shake(img, shift=5):
    h, w = img.shape[:2]
    tx = np.random.randint(-shift, shift)
    ty = np.random.randint(-shift, shift)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))


# ---------------------------------------------------------
# OCCLUSION & ROTATION
# ---------------------------------------------------------

def add_occlusion(img, max_area=0.15):
    h, w, _ = img.shape
    area = h * w
    occ_area = np.random.uniform(0.02, max_area) * area
    occ_w = int(np.sqrt(occ_area))
    occ_h = int(occ_w * np.random.uniform(0.5, 1.5))

    x = np.random.randint(0, w - occ_w)
    y = np.random.randint(0, h - occ_h)

    img2 = img.copy()
    color = tuple(np.random.randint(0, 60, size=3).tolist())
    cv2.rectangle(img2, (x, y), (x + occ_w, y + occ_h), color, -1)
    return img2


def rotate_scale(img, angle_range=25, scale_range=(0.8, 1.2)):
    angle = np.random.uniform(-angle_range, angle_range)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------
# MAIN RANDOM AUGMENT PIPELINE
# ---------------------------------------------------------

def random_augment(img):
    """Apply random real-world augmentations."""
    out = img.copy()

    # geometric
    if random.random() < 0.5:
        out = rotate_scale(out)
    if random.random() < 0.4:
        out = add_occlusion(out)

    # weather
    r = random.random()
    if r < 0.2:
        out = add_rain(out)
    elif r < 0.4:
        out = add_fog(out)
    elif r < 0.55:
        out = add_snow(out)

    # motion
    if random.random() < 0.3:
        out = motion_blur(out, degree=np.random.randint(5, 15), angle=np.random.randint(-10, 10))
    if random.random() < 0.3:
        out = camera_shake(out)

    return out


# ---------------------------------------------------------
# TF.DATA DATASET PIPELINE
# ---------------------------------------------------------

def load_dataset_paths(data_dir):
    """Returns list of (image_path, class_id)."""
    items = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        try:
            label = int(class_name)
        except:
            continue

        for file in os.listdir(class_dir):
            if file.lower().endswith((".png", ".ppm", ".jpg", ".jpeg")):
                items.append((os.path.join(class_dir, file), label))

    return items


def preprocess(path, label, augment=False):
    """Used inside tf.data pipeline."""
    def _py_func(path):
    # Convert EagerTensor → Python string
        path = path.numpy().decode("utf-8")
        img = read_image(path)
        return img


        if augment:
            img = random_augment(img)

        img = normalize(img)
        return img

    img = tf.py_function(_py_func, [path], tf.float32)
    img.set_shape((224, 224, 3))

    label = tf.one_hot(label, 43)
    return img, label


def create_dataset(data_list, batch_size=16, augment=False):
    paths = [p for p, _ in data_list]
    labels = [l for _, l in data_list]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=5000)
    ds = ds.map(lambda p, l: preprocess(p, l, augment), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds
