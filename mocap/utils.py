import os
import yaml
import threading

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pypylon import pylon
from sklearn.metrics import r2_score
from .imagezmq import ImageHub


# ################################################################################################################################
# Constants
# ################################################################################################################################


CAMERAS_NB = 4
CAMERA_IDXS = [127, 135, 143, 151]

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


# ################################################################################################################################
# Classes
# ################################################################################################################################


class SampleImageEventHandler(pylon.ImageEventHandler):
    
    def __init__(self, pipe_out, shared_memory, im_width: int, im_height: int, resize_en: bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe_out = pipe_out
        self.shared_memory = shared_memory
        self.shared_memory_np = np.frombuffer(self.shared_memory.get_obj(), dtype=np.uint8)
        
        self.resize_en = resize_en
        self.im_width  = im_width
        self.im_height = im_height
    
    def OnImageGrabbed(self, camera, grabResult):
        if grabResult.GrabSucceeded():
            img = grabResult.GetArray()
            with self.shared_memory.get_lock():
                if self.resize_en:
                    resized = cv2.resize(img, (self.im_width, self.im_height))
                    self.shared_memory_np[:] = resized.reshape(-1)
                else:
                    self.shared_memory_np[:] = img.reshape(-1)
            self.pipe_out.send("OK")


class ImageSaver():
    """ Simple class for saving images to a temporary folder, can be used for calibration/debuging """
    
    def __init__(self, dir: str, nb: int, step: int, clr: bool=True):
        if clr:
            self.dir = Path(dir).joinpath("tmp")
        else:
            dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.dir = Path(dir).joinpath(dt_string)
        if not self.dir.exists():
            self.dir.mkdir(parents=True)
        self.nb      = nb
        self.step    = step
        self.current = 0
        self.saved   = 0
        self.clear   = clr
    
    def save_image(self, img: np.ndarray) -> bool:
        if self.saved >= self.nb:
            return True
        if self.current % self.step == 0:
            img_name = f"frame_{self.saved:03}.bmp"
            cv2.imwrite(str(self.dir / img_name), img)
            print(f"ImageSaver::save_image: Saving frame no. [{self.saved}] to [{str(self.dir / img_name)}]")
            self.saved += 1
        self.current += 1
        return False
    
    def __del__(self):
        if self.clear:
            for file in self.dir.glob("*"):
                os.remove(file)
            self.dir.rmdir()


# Helper class implementing an IO deamon thread
class VideoStreamSubscriber:

    def __init__(self, hostnames: list, port: str, mode: int=1):
        self.hostnames = hostnames
        self.port = port
        self.mode = mode
        
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        
        print(f"VideoStreamSubscriber::__init__: Receiving from: [{self.hostnames[0]}:{self.port}]")
        self._thread.start()

    def receive(self, timeout: float=15.0) -> tuple:
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError("VideoStreamSubscriber::receive::ERROR: Timeout while waiting for publisher")
        self._data_ready.clear()
        return self._data

    def _run(self) -> None:
        receiver = ImageHub("tcp://{}:{}".format(self.hostnames[0], self.port), REQ_REP=False)
        for pub in self.hostnames[1:]:
            receiver.connect(f"tcp://{pub}:{self.port}")
        mode = self.mode
        rec_fun = None
        if mode == 0: rec_fun = receiver.recv_jpg
        elif mode == 1: rec_fun = receiver.recv_image
        while not self._stop:
            self._data = rec_fun()
            self._data_ready.set()
        receiver.close()

    def close(self) -> None:
        self._stop = True
        print(f"VideoStreamSubscriber::close: Stop receiving from [{self.hostnames[0]}:{self.port}]")


# ################################################################################################################################
# Functions (Edge device)
# ################################################################################################################################


def read_config(config_file: str) -> dict:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_chessborad_points(rows: int=6, cols: int=9) -> np.ndarray:
    points = np.zeros((rows*cols,3), np.float32)
    points[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
    return points


def calibrate_camera(imgs_path: str, config: dict, vis: bool=False) -> tuple:
    
    # Get chessboard shape
    ROWS, COLS = config["ROWS"], config["COLS"]
    
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ROWS*COLS,3), np.float32)
    objp[:,:2] = np.mgrid[0:COLS,0:ROWS].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Extract images from the directory 
    imgs_path = Path(imgs_path)
    
    # Find all images
    imgs = imgs_path.glob("*.bmp")
    for frame in imgs:
        img = cv2.imread(str(frame), -1)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (COLS,ROWS), None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            if vis:
                cv2.drawChessboardCorners(img, (COLS,ROWS), corners2, ret)
                cv2.imshow("Corners", img)
                cv2.waitKey(500)
    
    if vis: cv2.destroyWindow("Corners")
    calib_res = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    return calib_res


def move_pattern(pattern: dict, v: list=None, x: float=None, y: float=None, z: float=None):
    if v is None or len(v) != 3:
        if x is None: x = 0.0
        if y is None: y = 0.0
        if z is None: z = 0.0
        v = [x, y, z]
    new_pattern = {}
    for idx, coor in pattern.items():
        new_pattern[idx] = [coor[0]+v[0], coor[1]+v[1], coor[2]+v[2]]
    return new_pattern


def get_homography_matrix(image: np.ndarray, markers: list, pattern: dict, vis: bool=False):

    indexed_markers = [(0, x, y) for x, y, _ in markers]
    sorted_markers = find_corresponding_points(indexed_markers)
    
    # if vis:
    #     image_copy = image.copy()
    #     for idx, x, y in sorted_markers:
    #         cv2.circle(image_copy, (int(x), int(y)), 5, (255,0,0), 4)
    #         cv2.putText(image_copy, f"{idx+1}", (int(x)+10, int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    #     plt.imshow(image_copy)
    #     plt.show()

    img_points = np.array([[x, y, 0.] for _, x, y in sorted_markers], dtype='double')
    obj_points = np.array([pattern[idx][0:2] + [0.]  for idx, _, _ in sorted_markers], dtype='double')

    H, _ = cv2.findHomography(img_points, obj_points, method=cv2.RANSAC)
    return H


def find_corresponding_points(vec) -> list:
    """Sort points based on marker shape"""
    pairs = []
    r2_vec = []
    z_vec = []
    correct_idx = []

    # Make pairs of 3 elements, fit linear regression and calculate r2 score
    for i in range(6):
        for j in range(i, 6):
            for k in range(j, 6):
                if i != j and i != k and j != k:
                    x = np.array([vec[i][1], vec[j][1], vec[k][1]])
                    y = np.array([vec[i][2], vec[j][2], vec[k][2]])

                    z = np.polyfit(x, y, 1)
                    r2_vec.append(r2_score(y, z[0] * x + z[1]))
                    pairs.append([i, j, k])
                    z_vec.append(z)

    # Get first and second pair and parameters of linear regression based on best r2 score metric
    first_pair = pairs.pop(np.argmax(r2_vec))
    z_first = z_vec.pop(np.argmax(r2_vec))

    r2_vec.pop(np.argmax(r2_vec))

    second_pair = pairs.pop(np.argmax(r2_vec))
    z_second = z_vec.pop(np.argmax(r2_vec))

    # intersection is point of intersection of two straight lines
    intersection = [value for value in first_pair if value in second_pair]

    # separate is point that has no straight line
    separate = [value for value in [0, 1, 2, 3, 4, 5] if value not in first_pair + second_pair]

    # check if first pair of points belong to left or right side
    side = check_first_pair_side(vec[intersection[0]][1:], vec[separate[0]][1:], z_first, z_second)

    # append intersection and separate point to results
    correct_idx.append((0, vec[intersection[0]][1], vec[intersection[0]][2]))
    correct_idx.append((3, vec[separate[0]][1], vec[separate[0]][2]))

    if side:
        # side == 1 -> first pair belong to left side, second pair belong to right side
        left_pair = first_pair
        right_pair = second_pair
    else:
        # side == 0 -> second pair belong to left side, first pair belong to right side
        left_pair = second_pair
        right_pair = first_pair

    # For right side pair calculate which point is closer to intersection point
    right_pair.remove(intersection[0])

    idx1 = right_pair.pop()
    distance_1 = np.sqrt((vec[intersection[0]][1] - vec[idx1][1]) ** 2 + (vec[intersection[0]][2] - vec[idx1][2]) ** 2)

    idx2 = right_pair.pop()
    distance_2 = np.sqrt((vec[intersection[0]][1] - vec[idx2][1]) ** 2 + (vec[intersection[0]][2] - vec[idx2][2]) ** 2)

    if distance_1 > distance_2:
        correct_idx.append((1, vec[idx2][1], vec[idx2][2]))
        correct_idx.append((2, vec[idx1][1], vec[idx1][2]))
    else:
        correct_idx.append((2, vec[idx2][1], vec[idx2][2]))
        correct_idx.append((1, vec[idx1][1], vec[idx1][2]))

    # For left side pair calculate which point is closer to intersection point

    left_pair.remove(intersection[0])

    idx1 = left_pair.pop()
    distance_1 = np.sqrt((vec[intersection[0]][1] - vec[idx1][1]) ** 2 + (vec[intersection[0]][2] - vec[idx1][2]) ** 2)

    idx2 = left_pair.pop()
    distance_2 = np.sqrt((vec[intersection[0]][1] - vec[idx2][1]) ** 2 + (vec[intersection[0]][2] - vec[idx2][2]) ** 2)

    if distance_1 > distance_2:
        correct_idx.append((5, vec[idx2][1], vec[idx2][2]))
        correct_idx.append((4, vec[idx1][1], vec[idx1][2]))
    else:
        correct_idx.append((4, vec[idx2][1], vec[idx2][2]))
        correct_idx.append((5, vec[idx1][1], vec[idx1][2]))

    # Sort values based on index
    return sorted(correct_idx, key=lambda tup: tup[0])


def check_first_pair_side(intersection, separate, z_first, z_second) -> bool:
    """Return True if first pair is left side"""
    """Return False if first pair is right side"""

    x1, y1 = intersection
    x2, y2 = separate
    if z_first[0] > 0 and z_second[0] > 0:
        if z_first[0]*x2 + z_first[1] > y2:
            return True
        else:
            return False
    elif z_first[0] < 0 and z_second[0] < 0:
        if z_first[0]*x2 + z_first[1] > y2:
            return False
        else:
            return True
    else:
        if z_first[0]*x2 + z_first[1] > y2 and z_second[0]*x2 + z_second[1] > y2:
            if z_first[0] > 0:
                return True
            else:
                return False
        elif z_first[0]*x2 + z_first[1] < y2 and z_second[0]*x2 + z_second[1] < y2:

            if z_first[0] > 0:
                return True
            else:
                return False

        elif z_first[0]*x2 + z_first[1] < y2 and  z_first[0] < 0:
            return True
        elif z_first[0]*x2 + z_first[1] < y2 and  z_first[0] > 0:
            return False
        elif z_first[0]*x2 + z_first[1] > y2 and  z_first[0] < 0:
            return True
        elif z_first[0] * x2 + z_first[1] > y2 and z_first[0] > 0:
            return False


def detect_markers(image: np.ndarray, config: dict) -> list:
    """ Detect markers on the image """
    
    binary_threshold = config["BIN_THR"]
    kernel_size      = config["K_SIZE"]
    circ_threshold   = config["CIRC_THR"]
    w_h_difference   = config["W_H_DIFF"]
    marker_size      = config["MARK_SIZE"]
    area_threshold   = config["AREA_THR"]
    
    height, width = image.shape[0], image.shape[1]
    # gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(image, binary_threshold, 255, cv2.THRESH_BINARY) # +cv2.THRESH_OTSU
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # morph_image = binary_image
    
    # _, binary_image = cv2.threshold(gimg, 0, 40, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # +cv2.THRESH_OTSU
    # morph_image = binary_image

    connectivity = 8
    num_labels, _, stats, centroids =  cv2.connectedComponentsWithStats(morph_image, connectivity, cv2.CV_32S)
    objs = []
    
    for idx in range(1, num_labels):
        m_xi, m_yi = centroids[idx]
        m_width  = stats[idx, cv2.CC_STAT_WIDTH]
        m_height = stats[idx, cv2.CC_STAT_HEIGHT]
        m_radius = (m_width + m_height) / 4
        m_area   = stats[idx, cv2.CC_STAT_AREA]
        m_circularity = np.pi * m_radius**2/(m_area)

        # if (m_area >= area_threshold and 
        #     m_circularity > circ_threshold and 
        #     abs(m_width - m_height) <= w_h_difference and
        #     m_xi-marker_size >= 0 and m_xi+marker_size <= width and
        #     m_yi-marker_size >= 0 and m_yi+marker_size <= height):

        #     objs.append([*centroids[idx], m_radius])
        
        objs.append([*centroids[idx], m_radius])
    
    return objs
