import yaml

import cv2
import numpy as np


CAMERAS_NB = 4
CAMERA_IDS = [127, 135, 143, 151]

K_DEFAULT = np.array([[162,   0, 720],
                      [  0, 162, 540],
                      [  0,   0,   1]], dtype="double")

CHESSBOARD_LAYOUT = (6, 9)
CHESSBOARD_CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

PATTERN_PNB = 6
PATTERN = {
    0: [0.000, 0.000, 0.0],
    1: [0.175, 0.000, 0.0],
    2: [0.350, 0.000, 0.0],
    3: [0.350, 0.350, 0.0],
    4: [0.000, 0.350, 0.0],
    5: [0.000, 0.175, 0.0],
}

CMD_LEN = 8
CMD_OFFSET_EDGE = 8
CMD_OFFSET_SERVER = 16


def read_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_chessborad_points(rows: int=6, cols: int=9) -> np.ndarray:
    points = np.zeros((rows*cols,3), np.float32)
    points[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
    return points


def move_pattern(pattern: dict, v: list=None, x: float=None, y: float=None, z: float=None) -> dict:
    if v is None or len(v) != 3:
        if x is None: x = 0.0
        if y is None: y = 0.0
        if z is None: z = 0.0
        v = [x, y, z]
    new_pattern = {}
    for idx, coor in pattern.items():
        new_pattern[idx] = [coor[0]+v[0], coor[1]+v[1], coor[2]+v[2]]
    return new_pattern


def dict_to_indexed(struct: dict) -> list:
    res = []
    for key, item in struct.items():
        res.append((key, *item))
    return res


def indexes_to_dict(struct: list) -> dict:
    res = {}
    for key, *item in struct:
        res[key] = list(item[:])
    return res
