import time
import ctypes
import socket

from multiprocessing import Process, Array, Pipe

import cv2
import numpy as np
from pypylon import pylon

from .header import (
    PATTERN,
    PATTERN_PNB,
    CMD_LEN,
    CMD_OFFSET_EDGE,
    read_config,
    move_pattern,
)
from .imagezmq import ImageSender, ImageHub
from .utils import (
    SampleImageEventHandler,
    ImageSaver,
    detect_by_markers,
    find_corresponding_points,
    get_homography_matrix,
    get_projection_matrix,
)


class MocapCamera():
    
    def __init__(self, config_file: str, network_connected: bool=False):
        
        # Read configuration
        self.config = read_config(config_file)
        self.cmd_range = [CMD_OFFSET_EDGE, CMD_OFFSET_EDGE+CMD_LEN-1]

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
        self.im_width = self.camera.Width.GetValue()
        self.im_height = self.camera.Height.GetValue()
        self.camera.Close()
        
        # Create shared memory
        shared_array_size = self.im_width * self.im_height
        self.shared_memory = Array(ctypes.c_ubyte, shared_array_size)
        
        # Create a pipe for image passing
        self.camera_pipe = Pipe()
        
        # Create blocker flag
        self.blocked = False
        self.process = None
        
        # Create class for frame saving
        self.imgSaver = ImageSaver(
            dir =self.config["EDGE"].get("DIR", "edge/"),
            nb  =self.config["EDGE"].get("NB", 0),
            step=self.config["EDGE"].get("STEP", 1),
            clr =self.config["EDGE"].get("CLEAR", True),
        )
        
        # Get detection mode
        self.detection_mode = self.config["DETECTION"].get("MODE", 0)
    
    def run_sending(self):
        if self.blocked:
            print("MocapCamera::run_sending::WARNING: One process is already running - cannot start another one!")
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
        self._run()

    def run_detection(self):
        if self.blocked:
            print("MocapCamera::run_detection::WARNING: One process is already running - cannot start another one!")
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
        self._run()
    
    def run_detection_and_sending(self):
        if self.blocked:
            print("MocapCamera::run_detection::WARNING: One process is already running - cannot start another one!")
            return
        self.process = Process(
            target=_mp_send_detections_and_images, 
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config,
                  self.ip_addr)
        )
        self._run()
    
    def _run_dummy(self):
        if self.blocked:
            print("MocapCamera::run_calibration::WARNING: One process is already running - cannot start another one!")
            return
        self.process = Process(
            target=_mp_dummy_proc,
            args=(self.camera_pipe,
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config)
        )
        self._run()
    
    def _run(self):
        self.blocked = True 
        self.camera.Open()
        self._setup_camera()
        self._start_camera()
        self.process.start()
    
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
        
        if ext_trigger: 
            self.camera.RegisterConfiguration(
                pylon.SoftwareTriggerConfiguration(),
                pylon.RegistrationMode_ReplaceAll,
                pylon.Cleanup_Delete,
            )
        self.camera.RegisterImageEventHandler(
            SampleImageEventHandler(self.camera_pipe[0], self.shared_memory, self.im_width, self.im_height), 
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
            if msg == controller_ip and self.cmd_range[0] <= command[0] <= self.cmd_range[1]:
                code = command[0] - self.cmd_range[0]
                
                if code == 0:
                    if self.blocked: self.stop()
                    print("MocapCamera::listen: Subprocess: STOP")
                elif code == 1:
                    print("MocapCamera::listen: Run: [run_sending]")
                    self.run_sending()
                elif code == 2:
                    print("MocapCamera::listen: Run: [run_detection]")
                    self.run_detection()
                elif code == 3:
                    print("MocapCamera::listen: Run: [run_detection_and_sending]")
                    self.run_detection_and_sending()
                elif code == 7:
                    if self.blocked: self.stop()
                    print("MocapCamera::listen: Subprocess: STOP")
                    print("MocapCamera::listen: Program: STOP")
                    break
                else:
                    print(f"MocapCamera::listen::WARNING: Unknown command: [{code}]")
    
    def _calibrate_camera(self):
        self.camera.Open()
        self._setup_camera()
        self._start_camera()
        shared_np = np.frombuffer(self.shared_memory.get_obj(), dtype=np.uint8)
        scale = 0.5
        flag = False
        while True:
            status = self.camera[1].recv()
            if status == "STOP": break
            with self.shared_memory.get_lock():
                frame = np.copy(shared_np)
                frame = frame.reshape((self.im_height, self.im_width))
            cv2.imshow("frame", cv2.resize(frame, None, fx=scale, fy=scale))
            res = cv2.waitKey(1)
            if res == ord('q'):
                print(f"MocapCamera::SUBPROCESS::_calibrate_camera: Calibrate: STOP")
                break
            elif res == ord('s'):
                flag = self.imgSaver.save_image(frame)
                print(f"MocapCamera::SUBPROCESS::_calibrate_camera: Frame saved") 
            if flag:
                print(f"MocapCamera::SUBPROCESS::_calibrate_camera: Maximum number of images reached")
                break
        cv2.destroyWindow("frame")
        self.stop()
    
    def _compute_homography_matrix(self):
        self.camera.Open()
        self._setup_camera()
        self._start_camera()
        shared_np = np.frombuffer(self.shared_memory.get_obj(), dtype=np.uint8)
        new_pattern = move_pattern(PATTERN, v=[0., 0., 0.])
        scale = 0.5
        flag_en = False
        cam_id = self.ip_addr.split(".")[-1]
        while True:
            status = self.camera_pipe[1].recv()
            if status == "STOP": break
            with self.shared_memory.get_lock():
                frame = np.copy(shared_np)
                frame = frame.reshape((self.im_height, self.im_width))
            objs = detect_by_markers(frame, self.config["DETECTION"])
            flag_en = (len(objs) == PATTERN_PNB)
            if flag_en:
                indexed_markers = [(0, x, y) for x, y, _ in objs]
                sorted_markers = find_corresponding_points(indexed_markers)
            blank = np.zeros((self.im_height, self.im_width, 3), dtype=np.uint8)
            for i in [0,1,2]: blank[:,:,i] = frame
            for x, y, r in objs:
                cv2.circle(blank, (int(x), int(y)), int(r*2), (0,0,255), 2)
            if flag_en:
                for idx, x, y in sorted_markers:
                    cv2.putText(blank, f"{idx+1}", (int(x)+10, int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            blank = cv2.resize(blank, None, fx=scale, fy=scale)
            print(f"{np.array(objs).shape}")
            cv2.imshow("frame", blank)
            res = cv2.waitKey(1)
            if res == ord('s') and flag_en:
                H = get_homography_matrix(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), objs, new_pattern, True)
                np.save(f"edge/H_{cam_id}.npy", H)
                print(f"SAVED FILE TO [edge/H_{cam_id}.npy]")
            elif res == ord('q'):
                break
        self.stop()
    
    def _compute_projection_matrix(self):
        self.camera.Open()
        self._setup_camera()
        self._start_camera()
        shared_np = np.frombuffer(self.shared_memory.get_obj(), dtype=np.uint8)
        new_pattern = move_pattern(PATTERN, v=[0., 0., 0.])
        scale = 0.5
        flag_en = False
        cam_id = self.ip_addr.split(".")[-1]
        while True:
            status = self.camera_pipe[1].recv()
            if status == "STOP": break
            with self.shared_memory.get_lock():
                frame = np.copy(shared_np)
                frame = frame.reshape((self.im_height, self.im_width))
            objs = detect_by_markers(frame, self.config["DETECTION"])
            flag_en = (len(objs) == PATTERN_PNB)
            if flag_en:
                indexed_markers = [(0, x, y) for x, y, _ in objs]
                sorted_markers = find_corresponding_points(indexed_markers)
            blank = np.zeros((self.im_height, self.im_width, 3), dtype=np.uint8)
            for i in [0,1,2]: blank[:,:,i] = frame
            for x, y, r in objs:
                cv2.circle(blank, (int(x), int(y)), int(r*2), (0,0,255), 2)
            if flag_en:
                for idx, x, y in sorted_markers:
                    cv2.putText(blank, f"{idx+1}", (int(x)+10, int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            blank = cv2.resize(blank, None, fx=scale, fy=scale)
            print(f"{np.array(objs).shape}")
            cv2.imshow("frame", blank)
            res = cv2.waitKey(1)
            if res == ord('s') and flag_en:
                P = get_projection_matrix(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), objs, new_pattern, True)
                np.save(f"edge/P_{cam_id}.npy", P)
                print(f"SAVED FILE TO [edge/P_{cam_id}.npy]")
            elif res == ord('q'):
                break
        self.stop()


def _mp_send_images(
    pipe_out,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str,
) -> None:
    
    """ Send images to the server """
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    pub_port = config["PUBLISHERS"]["PORT_IMG"][0]
    sender = ImageSender(connect_to=f"tcp://*:{pub_port}", REQ_REP=False)
    print(f"MocapCamera::SUBPROCESS::_mp_send_images: Start processing")
    while True:
        status = pipe_out.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        _, frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sender.send_jpg_pubsub(f"{msg}", frame)
    return


def _mp_send_detections(
    pipe_out,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str,
) -> None:
    
    """ Thread used for marker detection """

    detection_mode = config["DETECTION"].get("MODE", 0)
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    pub_port = config["PUBLISHERS"]["PORT_PTS"][0]
    sender = ImageSender(connect_to=f"tcp://*:{pub_port}", REQ_REP=False)
    print(f"MocapCamera::SUBPROCESS::_mp_send_images: Start processing")
    while True:
        status = pipe_out.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        if detection_mode == 0:
            objs = detect_by_markers(frame, config["DETECTION"]["BY_MARKERS"])
        elif detection_mode == 1:
            objs = None
        sender.send_image_pubsub(f"{msg}", np.array(objs, dtype=np.float32))
    return


def _mp_send_detections_and_images(
    pipe_out,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str,
) -> None:
    
    """ Thread used for marker detection """
    
    detection_mode = config["DETECTION"].get("MODE", 0)
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    pub_port_img = config["PUBLISHERS"]["PORT_IMG"][0]
    sender_img = ImageSender(connect_to=f"tcp://*:{pub_port_img}", REQ_REP=False)
    pub_port_det = config["PUBLISHERS"]["PORT_PTS"][0]
    sender_det = ImageSender(connect_to=f"tcp://*:{pub_port_det}", REQ_REP=False)
    print(f"MocapCamera::SUBPROCESS::_mp_send_images: Start processing")
    while True:
        status = pipe_out.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        if detection_mode == 0:
            objs = detect_by_markers(frame, config["DETECTION"]["BY_MARKERS"])
        elif detection_mode == 1:
            objs = None
        _, frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sender_img.send_jpg_pubsub(f"{msg}", frame)
        sender_det.send_image_pubsub(f"{msg}", np.array(objs, dtype=np.float32))
    return


def _mp_dummy_proc(
    pipe,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    mode: int=0,
    resize_en: bool=False,
    wait_ns: float=16.0,
    msg: str="127.0.0.1",
) -> None:
    
    print(f"MocapCamera::SUBPROCESS::_mp_dummy: Init process")
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    cnt = 0
    
    # Mode 0.: Play the role of camera and create frames toprocess
    if mode == 0:
        
        blank = np.zeros((im_height, im_width), dtype=np.uint8)
        while True:
            image = blank.copy()
            cv2.putText(image, f"FRAME: {cnt}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cnt += 1
            
            with shared_memory.get_lock():
                if resize_en:
                    resized = cv2.resize(image, (im_width, im_height))
                    shared_np[:] = resized.reshape(-1)
                else:
                    shared_np[:] = image.reshape(-1)
            pipe[0].send("OK")
            
            cv2.imshow("dummy", image)
            if cv2.waitKey(wait_ns) == ord('q'):
                print(f"MocapCamera::SUBPROCESS::_mp_dummy: Dummy: STOP")
                break

    # Mode 1.: Receive frames and do processing (use for debuging locally)
    elif mode == 1:
        
        scale = 0.5
        cam_id = 0
        while True:
            status = pipe[1].recv()
            if status == "STOP": break
            with shared_memory.get_lock():
                frame = np.copy(shared_np)
                frame = frame.reshape((im_height, im_width))

            # # Display image from camera
            # frame_copy = frame.copy()
            # cv2.putText(frame_copy, f"FRAME: {cnt}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            # cnt += 1
            # print(f"MocapCamera::SUBPROCESS::_mp_dummy: Frame no. [{cnt}]")
            # cv2.imshow("dummy", frame_copy)
            # req = cv2.waitKey(1)
            
            # # Display detections on image
            # objs = detect_markers(frame, config["DETECTION"])
            # blank = np.zeros((im_height, im_width, 3), dtype=np.uint8)
            # for i in [0,1,2]: blank[:,:,i] = frame[:, :]
            # for x, y, r in objs:
            #     cv2.circle(blank, (int(x), int(y)), int(r*2), (0,0,255), 2)
            # blank = cv2.resize(blank, None, fx=scale, fy=scale)
            # print(f"{np.array(objs).shape}")
            # cv2.imshow("dummy", blank)
            # req = cv2.waitKey(1)
            
            # # Save detections
            # cnt += 1
            # plist = []
            # if req == ord('s'):
            #     plist.append(objs)
            #     print("MocapCamera::SUBPROCESS::_mp_dummy: Points saved")
            # if req == ord('f'):
            #     to_save = np.array(plist)
            #     np.save(f"recorded_data/detections/det_{cam_id}.npy", to_save)
            # if req == ord('i'):
            #     cv2.imwrite(f"recorded_data/im_{cam_id}_{cnt:3d}.jpg", frame)
            #     cnt += 1
            #     print("MocapCamera::SUBPROCESS::_mp_dummy: Frame saved")
    
    # Mode 1.: Receive frames and do processing (and send results)
    elif mode == 2:
        
        pub_port = config["PUBLISHERS"]["PORT"][0]
        sender = ImageSender(connect_to=f"tcp://*:{pub_port}", REQ_REP=False)
        while True:
            status = pipe[1].recv()
            if status == "STOP": break
            with shared_memory.get_lock():
                frame = np.copy(shared_np)
                frame = frame.reshape((im_height, im_width))
            
            # # Send image from camera
            # frame_copy = frame.copy()
            # frame_copy = cv2.putText(frame_copy, f"FRAME: {cnt}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            # cnt += 1
            # print(f"MocapCamera::SUBPROCESS::_mp_dummy: Frame no. [{cnt}]")
            # _, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # sender.send_jpg_pubsub(f"{msg}", encoded)
            
            # # Send detections
            # objs = detect_markers(frame, config["DETECTION"])
            # sender.send_image_pubsub(f"{msg}", np.array(objs, dtype=np.float32))
