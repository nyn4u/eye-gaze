"""Gaze extraction utilities (OpenCV-based with optional dlib fallback)

Primary class: GazeExtractor
Functions:
- extract_from_image(image_path)
- build_dataset_from_images(img_dir)
"""
import os
import glob
from typing import Tuple, List
import cv2
import numpy as np
import pandas as pd
import math

# try dlib (optional)
try:
    import dlib
    HAS_DLIB = True
except Exception:
    HAS_DLIB = False

HAAR_FACE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
HAAR_EYE = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(HAAR_FACE)
eye_cascade = cv2.CascadeClassifier(HAAR_EYE)


class GazeExtractor:
    def __init__(self, use_dlib=False, dlib_predictor_path=None):
        self.use_dlib = use_dlib and HAS_DLIB
        self.dlib_predictor_path = dlib_predictor_path
        if self.use_dlib and self.dlib_predictor_path is None:
            raise ValueError('dlib predictor path required if use_dlib=True')
        if self.use_dlib:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.dlib_predictor_path)

    def _detect_eyes_haar(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        eyes_found = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                eyes_found.append((x+ex, y+ey, ew, eh))
        return eyes_found

    def _eye_center_from_roi(self, roi: np.ndarray) -> Tuple[float, float]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = gray.shape
            return w/2.0, h/2.0
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M.get('m00', 0) == 0:
            h, w = gray.shape
            return w/2.0, h/2.0
        cx = M['m10']/M['m00']
        cy = M['m01']/M['m00']
        return cx, cy

    def extract_from_image(self, image_path: str) -> Tuple[float, float]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f'{image_path} not readable')
        h, w = img.shape[:2]
        if self.use_dlib:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = self.detector(gray)
            centers = []
            for d in dets:
                shape = self.predictor(gray, d)
                left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
                right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
                def mean_point(pts):
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    return (sum(xs)/len(xs), sum(ys)/len(ys))
                centers.append(mean_point(left_eye))
                centers.append(mean_point(right_eye))
            if centers:
                avgx = np.mean([c[0] for c in centers]) / w
                avgy = np.mean([c[1] for c in centers]) / h
                return float(avgx), float(avgy)
            # fallback
        eyes = self._detect_eyes_haar(img)
        if not eyes:
            return 0.5, 0.5
        centers = []
        for (ex, ey, ew, eh) in eyes:
            roi = img[ey:ey+eh, ex:ex+ew]
            cx, cy = self._eye_center_from_roi(roi)
            centers.append((ex + cx, ey + cy))
        avgx = np.mean([c[0] for c in centers]) / w
        avgy = np.mean([c[1] for c in centers]) / h
        return float(avgx), float(avgy)

def build_dataset_from_images(img_dir: str, recursive: bool = False, use_dlib: bool = False, dlib_path: str = None):
    pattern = '**/*.jpg' if recursive else '*.jpg'
    files = glob.glob(os.path.join(img_dir, pattern), recursive=recursive)
    files += glob.glob(os.path.join(img_dir, pattern.replace('jpg', 'png')), recursive=recursive)
    ge = GazeExtractor(use_dlib=use_dlib, dlib_predictor_path=dlib_path)
    rows = []
    for f in sorted(set(files)):
        try:
            gx, gy = ge.extract_from_image(f)
            rows.append({'image': os.path.basename(f), 'path': f, 'gaze_x': gx, 'gaze_y': gy})
        except Exception as e:
            print(f'Warning: {f} -> {e}')
    return pd.DataFrame(rows)
