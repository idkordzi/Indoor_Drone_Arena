import os
import ctypes

from multiprocessing import Process, Array, Pipe
from pathlib import Path
from datetime import datetime

from .header import read_config
from .server_functions import (
    new_process_start_receiving,
    new_process_start_processing,
)


class Server():
    def __init__(self, config_file: str, network_connected: bool=True, save_data: bool=False):
        
        # Read configuration
        self.config = read_config(config_file)
        
        publishers_nb = len(self.config["PUBLISHERS"]["IP"])
        
        self.im_width = self.config["CAMERA"]["IM_WIDTH"]
        self.im_height = self.config["CAMERA"]["IM_HEIGHT"]
        self.max_detections = self.config["DETECTION"]["MAX_DETECTIONS"]
        
        self.save_data  = save_data
        self.data_dir   = self.config["SERVER"].get("DIR", "server")
        self.data_clear = self.config["SERVER"].get("CLEAR", True)
        
        # Create directories for storing received data
        if self.save_data: self.exp_dir, self.pub_dirs = self._create_dirs()
        
        # Placeholder for server subprocesses
        self.process_receive = None
        self.process_process = None
        
        # Create communication pipes
        self.shared_memory = []
        for _ in range(publishers_nb):
            stream_mem = [
                Array(ctypes.c_ubyte, self.im_width * self.im_height),
                Array(ctypes.c_float, self.max_detections * 3),
                Array(ctypes.c_uint8, 1)
            ]
            self.shared_memory.append(stream_mem)
        
        # Create shared memory
        self.pipe_data_receive = Pipe()
        self.pipe_data_process = Pipe()

    def run(self, receive_mode: int=1) -> None:
        if self.process_receive is not None or self.process_process is not None:
            print("Server::run()::WARRNING: Subprocesses are already set!")
            return
        self.process_receive = Process(
            target=new_process_start_receiving, 
            args=(self.pipe_data_receive[0],
                  self.pipe_data_process[1],
                  self.shared_memory,
                  self.config,
                  receive_mode)
        )
        self.process_process = Process(
            target=new_process_start_processing, 
            args=(self.pipe_data_receive[1],
                  self.pipe_data_process[0],
                  self.shared_memory,
                  self.config,
                  receive_mode)
        )
        self.process_receive.start()
        self.process_process.start()
    
    def listen(self):
        raise NotImplementedError()
    
    def stop(self) -> None:
        print("Server::stop(): Stop receiving data")
        while self.pipe_data_receive[1].poll():
            self.pipe_data_receive[1].recv()
        self.pipe_data_receive[0].send("STOP")
        while self.pipe_data_process[1].poll():
            self.pipe_data_process[1].recv()
        self.pipe_data_process[0].send("STOP")
        print("Server::stop(): Waiting for process [1] to stop")
        if self.process_receive is not None:
            self.process_receive.join()
            self.process_receive.terminate()
        print("Server::stop(): Waiting for process [2] to stop")
        if self.process_process is not None:
            self.process_process.join()
            self.process_process.terminate()
        if self.save_data: self._rm_dirs()

    def _create_dirs(self) -> str:
        root = Path(self.data_dir)
        exp_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        exp_dir = root / exp_id
        pub_dirs = []
        for pub in self.publishers:
            pub_dir = exp_dir / pub
            pub_dir.mkdir(parents=True, exist_ok=True)
            pub_dirs.append(Path(pub_dir))
        return exp_dir, pub_dirs

    def _rm_dirs(self):
        if self.data_clear:
            for pub_dir in self.pub_dirs:
                for file in pub_dir.glob("*"):
                    os.remove(file)
                pub_dir.rmdir()
            self.exp_dir.rmdir()
