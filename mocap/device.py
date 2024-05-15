import ctypes
import socket

from multiprocessing import Process, Array, Pipe
from pypylon import pylon

from .header import read_config
from .utils import SampleImageEventHandler
from .device_functions import (
    new_process_send_images,
    new_process_send_detections,
    new_process_send_all,
    new_process_calibrate_camera,
    new_process_calculate_homography_matrix,
    new_process_calculate_projection_matrix,
    new_process_dummy_process,
)


class MocapCamera():
    
    def __init__(self, config_file: str, network_connected: bool=True, detection_method: int=1):
        
        # Read configuration file
        self.config = read_config(config_file)

        # Get device IP address
        self.network_connected = network_connected
        if self.network_connected:
            socket_handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            socket_handle.connect(("8.8.8.8", 80))
            self.ip_addr = socket_handle.getsockname()[0]
            socket_handle.shutdown(socket.SHUT_RDWR)
            socket_handle.close()
        else:
            hostname = socket.gethostname()
            self.ip_addr = socket.gethostbyname(hostname)
        print(f"MocapCamera::__init__::INFO: Connecting to [{self.ip_addr}]")

        # Create camera handler
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        
        # Get image shape
        self.im_width = self.camera.Width.GetValue()
        self.im_height = self.camera.Height.GetValue()
        print(f"MocapCamera::__init__::INFO: Found camera resolution [{self.im_width}] x [{self.im_height}]")
        self.camera.Close()
        
        # Create shared memory
        shared_array_size = self.im_width * self.im_height
        self.shared_memory = Array(ctypes.c_ubyte, shared_array_size)
        
        # Create a pipe for process communication
        self.camera_pipe = Pipe()
        
        # Create blocker flag
        self.blocked = False
        self.process = None
        
        # Get detection method # @TODO need fix
        # detection_mode = self.config["DETECTION"].get("METHOD", "MARKERS")
        if detection_method == 1:
            self.detection_mode = "MARKERS"
        elif detection_method == 2:
            self.detection_mode = "FOREGROUND"
        self.config["DETECTION"]["METHOD"] = self.detection_mode
        
    
    def listen(self):
        raise NotImplementedError()
    
    def run_process_send_images(self):
        if self.blocked:
            print("MocapCamera::run_process_send_images::WARNING: "
                  "One process is already running - cannot start another one!")
            return
        self.process = Process(
            target=new_process_send_images,
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config,
                  self.ip_addr)
        )
        self._run()

    def run_process_send_detections(self):
        if self.blocked:
            print("MocapCamera::run_process_send_detections::WARNING: "
                  "One process is already running - cannot start another one!")
            return
        self.process = Process(
            target=new_process_send_detections, 
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config,
                  self.ip_addr)
        )
        self._run()
    
    def run_process_send_all(self):
        if self.blocked:
            print("MocapCamera::run_process_send_all::WARNING: "
                  "One process is already running - cannot start another one!")
            return
        self.process = Process(
            target=new_process_send_all, 
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config,
                  self.ip_addr)
        )
        self._run()
    
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
    
    def _run(self):
        self.blocked = True 
        self.camera.Open()
        self._setup_camera()
        self._start_camera()
        self.process.start()
    
    def _setup_camera(self):
        
        # if external_trigger: 
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
        
        external_trigger = self.config["CAMERA"].get("TRIGGER", False) 
        if external_trigger:
            # self.camera.LineSelector.SetValue("Line3")
            # self.camera.LineMode.SetValue("Input")
            self.camera.TriggerSelector.SetValue("FrameStart")
            self.camera.TriggerSource.SetValue("Line3")
            self.camera.TriggerMode.SetValue("On")
            self.camera.TriggerActivation.SetValue("RisingEdge")
        else:
            fps = self.config["CAMERA"].get("FPS", 30)
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
            self.camera.AcquisitionFrameRate.SetValue(fps)
        exposure_time = self.config["CAMERA"].get("EXPOSURE", 6000.0)
        self.camera.ExposureTime.SetValue(exposure_time)
        self.camera.PixelFormat.SetValue("Mono8")
        self.camera.MaxNumBuffer.SetValue(10)
    
    def _start_camera(self):
        self.camera.StartGrabbing(
            pylon.GrabStrategy_LatestImageOnly,
            pylon.GrabLoop_ProvidedByInstantCamera,
        )
    
    def _stop_camera(self):
        self.camera.StopGrabbing()
    
    def _run_process_calibrate_camera(self):
        if self.blocked:
            print("MocapCamera::_run_process_calibrate_camera::WARNING: "
                  "One process is already running - cannot start another one!")
            return
        self.process = Process(
            target=new_process_calibrate_camera,
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config,
                  self.ip_addr)
        )
        self._run()
    
    def _run_process_calculate_homography_matrix(self):
        if self.blocked:
            print("MocapCamera::_run_process_calculate_homography_matrix::WARNING: "
                  "One process is already running - cannot start another one!")
            return
        self.process = Process(
            target=new_process_calculate_homography_matrix,
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config,
                  self.ip_addr)
        )
        self._run()
    
    def _run_process_calculate_projection_matrix(self):
        if self.blocked:
            print("MocapCamera::_run_process_calculate_projection_matrix::WARNING: "
                  "One process is already running - cannot start another one!")
            return
        self.process = Process(
            target=new_process_calculate_projection_matrix,
            args=(self.camera_pipe[1],
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config,
                  self.ip_addr)
        )
        self._run()

    def _run_dummy_process(self):
        if self.blocked:
            print("MocapCamera::_run_dummy_process::WARNING: "
                  "One process is already running - cannot start another one!")
            return
        self.process = Process(
            target=new_process_dummy_process,
            args=(self.camera_pipe,
                  self.shared_memory,
                  self.im_height,
                  self.im_width,
                  self.config)
        )
        self._run()
