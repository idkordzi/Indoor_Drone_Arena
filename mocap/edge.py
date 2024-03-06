import time
import ctypes
import socket

from multiprocessing import Process, Array, Pipe

import cv2
import numpy as np

from pypylon import pylon
from .imagezmq import ImageSender, ImageHub
from .utils import (
    PATTERN,
    SampleImageEventHandler,
    ImageSaver,
    read_config,
    detect_markers,
    move_pattern,
    get_homography_matrix,
)


class MocapCamera():
    
    def __init__(self, config_file: str, network_connected: bool=False):
        
        # Read configuration
        self.config = read_config(config_file)

        # Get device IP address
        self.connected = network_connected
        if self.connected:
            socket_handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            socket_handle.connect(("8.8.8.8", 80))
            self.ip_addr = socket_handle.getsockname()[0]
            socket_handle.shutdown(socket.SHUT_RDWR)
            socket_handle.close()
        else:
            hostname = socket.gethostname()
            self.ip_addr = socket.gethostbyname(hostname)

        # Create camera handler
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        
        # Get image shape
        c_im_width  = self.config["CAMERA"].get("IM_WIDTH", 1.0)
        c_im_height = self.config["CAMERA"].get("IM_HEIGHT", 1.0)
        
        if type(c_im_width) is int:
            self.im_width = c_im_width
        elif type(c_im_width) is float:
            self.im_width = int(self.camera.Width.GetValue() * c_im_width)
        if self.im_width > self.camera.Width.GetValue():
            print("MocapCamera::__init__::WARNING: Requested image width is greater than camera default")
        
        if type(c_im_height) is int:
            self.im_height = c_im_height
        elif type(c_im_height) is float:
            self.im_height = int(self.camera.Height.GetValue() * c_im_height)
        if self.im_height > self.camera.Height.GetValue():
            print("MocapCamera::__init__::WARNING: Requested image height is greater than camera default")
        
        if self.im_width == self.camera.Width.GetValue() and self.im_height == self.camera.Height.GetValue():
            self.resize_en = False
        else:
            self.resize_en = True
        
        # End camera configuration
        self.camera.Close()
        
        # Create shared memory
        shared_array_size = self.im_width * self.im_height
        self.shared_memory = Array(ctypes.c_ubyte, shared_array_size)
        
        # Create a pipe for image passing
        self.camera_pipe = Pipe()
        
        # Create blocker flag
        self.blocked = False
        self.process = None
    
    def run_sending(self):
        if self.blocked:
            print("MocapCamera::run_sending::WARNING: This process is already running - cannot start another one!")
            return
        
        self.process = Process(
            target=_mp_send_images,
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config,
                  self.ip_addr)
        )
        self.blocked = True
        
        self.camera.Open()
        self._setup_camera()
        self._start_camera()
        self.process.start()

    def run_detection(self):
        if self.blocked:
            print("MocapCamera::run_detection::WARNING: This process is already running - cannot start another one!")
            return
        
        self.process = Process(
            target=_mp_detect_markers, 
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config,
                  self.ip_addr)
        )
        self.blocked = True
        
        self.camera.Open()
        self._setup_camera()
        self._start_camera()
        self.process.start()
    
    def run_detection_and_sending(self):
        if self.blocked:
            print("MocapCamera::run_detection::WARNING: This process is already running - cannot start another one!")
            return
        
        self.process = Process(
            target=_mp_send_detections, 
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config,
                  self.ip_addr)
        )
        self.blocked = True
        
        self.camera.Open()
        self._setup_camera()
        self._start_camera()
        self.process.start()
    
    def run_calibration(self):
        if self.blocked:
            print("MocapCamera::run_calibration::WARNING: This process is already running - cannot start another one!")
            return
        
        self.process = Process(
            target=_mp_collect_calibration_data,
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config)
        )
        self.blocked = True 
        
        self.camera.Open()
        self._setup_camera()
        self._start_camera()
        self.process.start()
    
    def _calibrate_device(self):
        
        from .utils import find_corresponding_points
        
        self.camera.Open()
        self._setup_camera()
        self._start_camera()
        
        shared_np = np.frombuffer(self.shared_memory.get_obj(), dtype=np.uint8)
        new_pattern = move_pattern(PATTERN, v=[0., 0., 0.])
        
        cam_id = 127
        
        while True:
            status = self.camera_pipe[1].recv()
            if status == "STOP": break
            with self.shared_memory.get_lock():
                frame = np.copy(shared_np)
                frame = frame.reshape((self.im_height, self.im_width))
            objs = detect_markers(frame, self.config["DETECTION"])
            
            if len(objs) != 6:
                continue
            
            indexed_markers = [(0, x, y) for x, y, _ in objs]
            sorted_markers = find_corresponding_points(indexed_markers)
            
            blank = np.zeros((self.im_height, self.im_width, 3), dtype=np.uint8)
            for i in [0,1,2]: blank[:,:,i] = frame
            for x, y, r in objs:
                cv2.circle(blank, (int(x), int(y)), int(r*2), (0,0,255), 2)
            for idx, x, y in sorted_markers:
                cv2.putText(blank, f"{idx+1}", (int(x)+10, int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            blank = cv2.resize(blank, None, fx=0.5, fy=0.5)
            print(f"{np.array(objs).shape}")
            cv2.imshow("frame", blank)
            
            res = cv2.waitKey(1)
            if res == ord('s'):
                H = get_homography_matrix(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), objs, new_pattern, True)
                np.save(f"saved_data/H_{cam_id}.npy", H)
                print(f"SAVED FILE TO [saved_data/H_{cam_id}.npy]")
            elif res == ord('q'):
                break
        self.stop()
    
    def _setup_camera(self):
        
        ext_trigger = self.config["CAMERA"].get("EXT_TRIGGER", False) 
        if ext_trigger:
            # self.camera.LineSelector.SetValue("Line3")
            # self.camera.LineMode.SetValue("Input")
            # self.camera.TriggerSelector.SetValue("FrameStart")
            # self.camera.TriggerActivation.SetValue("RisingEdge")
            self.camera.TriggerMode.SetValue("On")
            self.camera.TriggerSource.SetValue("Line3")
        else:
            fps = self.config["CAMERA"].get("FPS", 30)
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
            self.camera.AcquisitionFrameRate.SetValue(fps)
        
        exposure_time = self.config["CAMERA"].get("EXPOSURE", 10000.0)
        self.camera.ExposureTime.SetValue(exposure_time)
        self.camera.PixelFormat.SetValue("Mono8")
        
        # Set camera events handling
        if ext_trigger: 
            self.camera.RegisterConfiguration(
                pylon.SoftwareTriggerConfiguration(),
                pylon.RegistrationMode_ReplaceAll,
                pylon.Cleanup_Delete,
            )
        
        self.camera.RegisterImageEventHandler(
            SampleImageEventHandler(self.camera_pipe[0], self.shared_memory, self.im_width, self.im_height, self.resize_en), 
            pylon.RegistrationMode_Append,
            pylon.Cleanup_Delete,
        )
    
    def _start_camera(self):
        self.camera.StartGrabbing(
            pylon.GrabStrategy_LatestImageOnly,
            pylon.GrabLoop_ProvidedByInstantCamera,
        )
    
    def _stop_camera(self):
        self.camera.StopGrabbing()
    
    def stop(self):
        self._stop_camera()
        self.camera.Close()
        while self.camera_pipe[1].poll():
            self.camera_pipe[1].recv()
        self.camera_pipe[0].send("STOP")
        if self.process is not None:
            print("MocapCamera::stop: Waiting for process to stop")
            self.process.join()
            self.process.terminate()
        self.blocked = False
    
    def listen(self):
        controller_ip   = self.config["CONTROLLER"]["IP"][0]
        controller_port = self.config["CONTROLLER"]["PORT"][0]
        self.listener = ImageHub(f"tcp://{controller_ip}:{controller_port}", REQ_REP=False)
        
        while True:
            msg, command = self.listener.recv_image()
            print(f"MocapCamera::listen: Receiving: [{msg}] {command}")
            if msg == controller_ip:
                
                if command == 0:
                    if self.blocked: self.stop()
                    print("MocapCamera::listen: Subprocess: STOP")
                    print("MocapCamera::listen: Program: STOP")
                    break
                elif command == 1:
                    if self.blocked: self.stop()
                    print("MocapCamera::listen: Subprocess: STOP")
                elif command == 2:
                    print("MocapCamera::listen: Run: [run_sending]")
                    self.run_sending()
                elif command == 3:
                    print("MocapCamera::listen: Run: [run_detection]")
                    self.run_detection()
                else:
                    print("MocapCamera::listen::WARNING: Unknown command")


def _mp_send_images(
    pipe_in,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str,
) -> None:
    
    """ Send images to the server """
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    pub_port = config["PUBLISHERS"]["PORT"][0]
    sender = ImageSender(connect_to=f"tcp://*:{pub_port}", REQ_REP=False)
    
    # DEBUG
    cnt = 0
    
    while True:
        status = pipe_in.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        
        # DEBUG
        cnt += 1
        frame = cv2.putText(frame, f"frame: {cnt}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        print(f"MocapCamera::SUBPROCESS::_mp_send_images: Sent frame no. [{cnt}]")
        cv2.waitKey(100)
        
        _, frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sender.send_jpg_pubsub(f"{msg}", frame)
    
    return


def _mp_detect_markers(
    pipe_in,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str,
) -> None:
    
    """ Thread used for marker detection """

    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    pub_port = config["PUBLISHERS"]["PORT"][0]
    sender = ImageSender(connect_to=f"tcp://*:{pub_port}", REQ_REP=False)
    
    # DEBUG
    plist = []
    cam_id = 127
    ct = 0
        
    while True:
        status = pipe_in.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        objs = detect_markers(frame, config["DETECTION"])
        
        blank = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        for i in [0,1,2]: blank[:,:,i] = frame
        for x, y, r in objs:
            cv2.circle(blank, (int(x), int(y)), int(r*2), (0,0,255), 2)
        blank = cv2.resize(blank, None, fx=0.5, fy=0.5)
        print(f"{np.array(objs).shape}")
        # for pt in objs:
        #     print(f"[{} {}]")
        cv2.imshow("frame", blank)
        
        res = cv2.waitKey(1)
        if res == ord('s'):
            plist.append(objs)
            print("PTS SAVED: OK")
        if res == ord('i'):
            cv2.imwrite(f"recorded_data/detections/im_{cam_id}_{ct}.bmp", frame)
            ct += 1
            print("IMG SAVED: OK")
        
        sender.send_image_pubsub(f"{msg}", np.array(objs, dtype=np.float32))
    
    to_save = np.array(plist)
    np.save(f"recorded_data/detections/det_{cam_id}.npy", to_save)
    
    return


def _mp_send_detections(
    pipe_in,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str,
) -> None:
    
    """ Thread used for marker detection """

    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    pub_port_det = config["PUBLISHERS"]["PORT"][0]
    sender_det = ImageSender(connect_to=f"tcp://*:{pub_port_det}", REQ_REP=False)
    pub_port_img = config["PUBLISHERS"]["PORT"][1]
    sender_img = ImageSender(connect_to=f"tcp://*:{pub_port_img}", REQ_REP=False)
        
    while True:
        status = pipe_in.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        objs = detect_markers(frame, config["DETECTION"])
        _, frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        sender_det.send_image_pubsub(f"{msg}", np.array(objs, dtype=np.float32))
        sender_img.send_jpg_pubsub(f"{msg}", frame)
    
    return


def _mp_collect_calibration_data(
    pipe_in,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
) -> None:
    
    """ Thread used for marker detection """
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    imgSaver = ImageSaver(
        dir =config["EDGE"].get("DIR", "recorded_data/"),
        nb  =config["EDGE"].get("NB", 0),
        step=config["EDGE"].get("STEP", 1),
        clr =config["EDGE"].get("CLEAR", True),
    )
    
    scale = 0.5
    flag = False
    while True:
        status = pipe_in.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        
        cv2.imshow("frame", cv2.resize(frame, None, fx=scale, fy=scale))
        res = cv2.waitKey(1)
        if res == ord('q'):
            print(f"MocapCamera::SUBPROCESS::_mp_collect_calibration_data: Subprocess: STOP")
            break
        elif res == ord('s'):
            flag = imgSaver.save_image(frame)
            print(f"MocapCamera::SUBPROCESS::_mp_collect_calibration_data: Frame saved") 
        if flag:
            print(f"MocapCamera::SUBPROCESS::_mp_collect_calibration_data: Maximum number of images reached")
            break
    
    cv2.destroyWindow("frame")
        
    return


def _mp_dummy_proc(
    pipe_out,
    shared_memory,
    im_height: int,
    im_width: int,
    resize_en: bool,
    wait_s: float=16.0,
) -> None:
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    
    # Create dummy image
    img = np.zeros((im_height, im_width), dtype=np.uint8)
    wait_ns = int(wait_s*1000)
    
    while True:
        with shared_memory.get_lock():
            if resize_en:
                resized = cv2.resize(img, (im_width, im_height))
                shared_np[:] = resized.reshape(-1)
            else:
                shared_np[:] = img.reshape(-1)
        pipe_out.send("OK")
        
        # time.sleep(wait_s)
        
        cv2.imshow("dummy", img)
        if cv2.waitKey(wait_ns) == ord('q'):
            print(f"MocapCamera::SUBPROCESS::_mp_dummy_proc: Dummy - STOP")
            break
