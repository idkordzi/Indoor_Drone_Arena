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


def dict_to_list(struct: dict):
    new_struct = []
    for key, item in struct.items():
        new_struct.append((key, *item))
    return new_struct


def list_to_dict(struct: list):
    new_struct = {}
    for key, *item in struct:
        new_struct[key] = item
    return new_struct


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


def detect_markers(image: np.ndarray, config: dict) -> list:
    """ Detect markers on the image """
    
    binary_threshold = config["BIN_THR"]
    kernel_size      = config["K_SIZE"]
    circ_threshold   = config["CIRC_THR"]
    w_h_difference   = config["W_H_DIFF"]
    marker_size      = config["MARK_SIZE"]
    area_threshold   = config["AREA_THR"]
    
    height, width = image.shape[0], image.shape[1]
    gimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gimg, binary_threshold, 255, cv2.THRESH_BINARY) # +cv2.THRESH_OTSU
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    # morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    morph_image = binary_image
    
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
    
    # try:
    #     g_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     print("CONVERT TO GRAYSCALE")
    # except:
    #     g_frame = image
    #     print("NO CONVERTION")
    # _, thr = cv2.threshold(g_frame, 0, 40, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # connectivity = 8
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thr, connectivity, cv2.CV_32S)
    # obj = []
    # for id, (x, y) in enumerate(centroids[1:]):
    #     obj.append((x, y, 0))

    # return obj


# ################################################################################################################################
# Functions (Marker indexing)
# ################################################################################################################################


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


# ################################################################################################################################
# Functions (Pose estimation)
# ################################################################################################################################


def compute_homography(img_points, obj_points):
    A = []
    for i in range(0, len(img_points)):
        u, v = img_points[i][0], img_points[i][1]
        x, y = obj_points[i][0], obj_points[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H


def camera_pose_from_homography(H, K):  
    H = np.linalg.inv(K)@H
    norm1 = np.linalg.norm(H[:, 0])
    norm2 = np.linalg.norm(H[:, 1])
    tnorm = (norm1 + norm2) / 2.0

    H1 = H[:, 0] / norm1
    H2 = H[:, 1] / norm2
    H3 = np.cross(H1, H2)
    T = H[:, 2] / tnorm
    P = np.array([H1, H2, H3, T]).transpose()
    return K @ P


def triangulate_nviews(P, ip):
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        x = np.append(x, 1)
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]


def find_camera_position(img_points, obj_points, K):
    """Find camera position by solving PnP problem for a coplanar marker"""
    img_points = np.float32(img_points)
    obj_points = np.float32(obj_points)

    # Asume no camera distortion
    dist_coeffs = np.zeros((4,1))
    
    success, r_vec, t_vec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    r_vec, t_vec = cv2.solvePnPRefineVVS(obj_points, img_points, K, dist_coeffs, r_vec, t_vec)

    r_mat = cv2.Rodrigues(r_vec)[0]

    return r_mat, t_vec


def calculate_projection_matrix(r_mat, t_vec, K):
    return K @ np.hstack([r_mat,t_vec])


def triangulate_points(P1, P2, p1, p2):
    point = cv2.triangulatePoints(P1, P2, p1, p2)
    point /= point[-1]

    return point[:-1]


def n_view_traingulation(P_vec, img_points):
    """
    Created this function based on webpage: 
    https://amytabb.com/tips/tutorials/2021/10/31/triangulation-DLT-2-3/
    P_vec - vector of P arrays for each camera
    img_points - corestornding point in the image
    """
    create_row = lambda u, v, P : np.vstack(( u*P[2,:]-P[0,:],
                                              v*P[2,:]-P[1,:]))

    A = np.vstack([create_row(u,v,P) for (u, v), P  in zip(img_points, P_vec)])
    
    # Solve the A*X = 0 using SVD
    u, s, vh = np.linalg.svd(A)
    X = vh[-1,:]

    return X/X[-1]


def calculate_points_in_space(cameras: dict):   
    estim_points = []
    for i, in range(len(cameras.keys())):
        points = []
        P_vec = []
        for ci, cam in cameras.items():
            P_vec.append(cam["P"])  
            point = (cam["markers"][i][1], cam["markers"][i][2])      
            points.append(point)

        rec = n_view_traingulation(P_vec, points)
        rec = rec[:-1]
        estim_points.append(rec)
    return estim_points


# ################################################################################################################################
# Functions (Evaluation and visualization)
# ################################################################################################################################


def draw_cube(img, P):
    """Function draw a unit qube on an image"""
    # cube = [[0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1], [1,0,1], [0,1,1], [1,1,1]]
    cube = [[0,0,0], [0.35,0,0], [0,0.35,0], [0.35,0.35,0], [0,0,0.35], [0.35,0,0.35], [0,0.35,0.35], [0.35,0.35,0.35]]
    cube_lines = lambda cube: [
        (cube[0], cube[1]),
        (cube[0], cube[2]),
        (cube[3], cube[1]),
        (cube[3], cube[2]),
        (cube[4], cube[5]),
        (cube[4], cube[6]),
        (cube[7], cube[5]),
        (cube[7], cube[6]),
        (cube[0], cube[4]),
        (cube[1], cube[5]),
        (cube[2], cube[6]),
        (cube[3], cube[7])
    ]

    # Translate the cube cooridinates to the iamge cooridnates
    img_cube = []
    for vec in cube:
        # Project the 3D point to 2D image
        proj = P @ np.append(vec, 1)
        # Normalize the vector
        proj /= proj[-1]
        img_cube.append(proj[:-1])

    img_lines = cube_lines(img_cube)

    # Read image and draw lines
    for line in img_lines:
        line = np.squeeze(line)
        img = cv2.line(img, line[0].astype(int), line[1].astype(int), (0,0,255), 2)

    return img


def plot_cameras_in_space(cameras, patter):
    """ Plot cameras with marker on a 3D plot"""
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for id, cam in enumerate(cameras):
        r_mat = cam["R"]
        t_vec = cam["t"]
        coord = -np.matrix(r_mat).T @ np.matrix(t_vec)

        ax.scatter3D(coord[0], coord[1], coord[2], label=f'cam: {id+1}')
    ax.set_aspect('auto')

    # Draw marker
    p = np.array(list(patter.values()))
    ax.scatter(p[:,0], p[:,1], p[:,2],marker='*',label='Markers')
    plt.legend()
    plt.show()


def calculate_error(img_points, obj_points, P):
    errors = []
    projections = []
    for img_p, obj_p in zip(img_points, obj_points):
        proj = P @ np.append(obj_p, [1.])
        proj /= proj[-1]
        error = (img_p - proj[:-1])
        projections.append(proj[:-1])
        errors.append(error)

    # Calcualte metrics

    # 1. Calculate sum of squred errors
    SSE = np.concatenate(np.power(errors,2)).sum()

    # 2. Display the SSE
    print(f"Sum of squred errors of reproductions is: {SSE}\n")

    # 3. Print all the points
    for img_p, proj, err in zip(img_points, projections, errors):
        print(f"point:\t\t{img_p}\nprojection:\t{proj}\nerror: \t\t{err}\n")


def get_pattern_inner_dims(pattern: list):
    res = []
    for idx, *pi in pattern:
        arr = []
        for _, *pj in pattern:
            arr.append(np.linalg.norm(np.array(pi) - np.array(pj)))
        res.append((idx, arr))
    return res


def get_inner_error_matrix(errors: list):
    matrix = []
    for idx, *item in errors:
        matrix.append(*item)
    return (np.array(matrix))


def calculate_inner_errors(est_points, obj_points):
    estimation = []
    pattern = []
    idx = 0
    for est_p, obj_p in zip(est_points, obj_points):
        estimation.append((idx, est_p))
        pattern.append((idx, obj_p))
        idx += 1
    est_dims = get_pattern_inner_dims(estimation)  
    patt_dims = get_pattern_inner_dims(pattern)
    
    error_matrix = get_inner_error_matrix(patt_dims) - get_inner_error_matrix(est_dims)
    return error_matrix
