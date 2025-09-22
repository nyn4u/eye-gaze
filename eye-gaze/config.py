# small config used by scripts
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMG_DIR = os.path.join(ROOT_DIR, 'images')
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, 'outputs')
DLIB_LANDMARK_PATH = os.path.join(ROOT_DIR, 'shape_predictor_68_face_landmarks.dat')
