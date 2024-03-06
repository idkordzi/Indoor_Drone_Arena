import os
import time
import threading

from copy import copy
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from .imagezmq import ImageSender
from .utils import CAMERA_IDXS, VideoStreamSubscriber, read_config


class Server():
    def __init__(self, config_file: str, mode: int=1, save_data: bool=False):
        
        # Read configuration
        self.config = read_config(config_file)
        self.publishers = self.config["PUBLISHERS"]["IP"]
        self.subs_port  = self.config["PUBLISHERS"]["PORT"]
        
        self.save_data  = save_data
        self.data_dir   = self.config["SERVER"].get("DIR", "received_data/")
        self.data_clear = self.config["SERVER"].get("CLEAR", True)
        
        self.timeout = self.config["SERVER"].get("TIMEOUT", 10.0)
        
        # Create directories for storing received data
        if self.save_data: self.exp_dir, self.pub_dirs = self._create_dirs()
        
        # Prepare for streams for receiving data
        self.mode = mode
        self.streams = []
        self.threads = []
        self.comms  = []
        self.buffer = []
        for p in self.publishers:
            self.streams.append(VideoStreamSubscriber([p], self.subs_port[0], self.mode))
            self.comms.append(threading.Event())
            self.buffer.append(["", None])
        
        # Prepare for main thread for processing data
        self.main_thread = None
        
        # Create threading events
        self.flags = []
        for _ in range(3): self.flags.append(threading.Event())

    def receive_images(self) -> None:
        try:
            # Set threads for receiving data
            for i in range(len(self.streams)):
                self.threads.append(threading.Thread(
                    target=run_stream,
                    args=(
                        self.streams[i],
                        self.buffer[i],
                        self.flags,
                        self.comms[i],
                        self.mode,
                        self.timeout
                    )
                ))
            for i in range(len(self.threads)):
                self.threads[i].start()
            
            # Set main thread for processing data
            self.main_thread = threading.Thread(
                target=run_processing,
                args=(
                    self.buffer,
                    self.flags,
                    self.comms,
                    self.mode
                )
            )
            self.main_thread.start()
            
            # Start program
            self.flags[2].set()
            self.flags[0].set()
        except:
            print("Server::receive_images::ERROR: Unknown error occured - stop receiving data")
            self.flags[1].set()
            if self.save_data: self._rm_dirs()
    
    def stop(self) -> None:
        print("Server::stop: Stop receiving data")
        self.flags[1].set()
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


def run_stream(
    stream: VideoStreamSubscriber,
    buffer: list,
    flags: list,
    ready: threading.Event,
    mode: int,
    timeout: float
) -> None:
    
    cnt = 0
    pub_addr = f"{stream.hostnames[0]}:{stream.port}"
    
    # Wait for signal from master
    while True:
        if flags[0].is_set(): break
    
    print(f"Server::THREAD::run_stream: Start receiving from [{pub_addr}]")
    
    # Main loop
    while True:
        
        if flags[1].is_set():
            break
        if not flags[2].is_set() or ready.is_set():
            continue
        
        # Receive data
        msg, data = stream.receive(timeout=timeout)
        buffer[0] = msg
        if mode == 0:
            buffer[1] = cv2.imdecode(np.frombuffer(data, dtype='uint8'), -1)
        elif mode == 1: buffer[1] = data.copy()
        
        cnt += 1
        ready.set()
    
    stream.close()


def run_processing(
    buffer: list,
    flags: list,
    comms: list,
    mode: int
) -> None:
    
    from .utils import find_corresponding_points
    
    flag_wait = False
    streams_nb = len(comms)
    local_mem = [["", None] for _ in range(streams_nb)]
    
    # Wait for signal from master
    while True:
        if flags[0].is_set(): break
    
    print(f"Server::THREAD::run_processing: Start processing")
    
    # TODO load H matrixes
    H_matrixes = {}
    for idx in CAMERA_IDXS:
        mat = np.load(f"saved_data/H_{idx}.npy")
        H_matrixes[idx] = mat.copy()
    
    # # DEBUG
    # av_time = 0.
    # cnt = 0
    #
    # # sq = np.ceil(np.sqrt(streams_nb))
    # # bg = np.zeros((360*sq, 480*sq), dtype=np.uint8)
    #
    # plist = []
    
    # Main loop
    while True:
        
        t0 = time.perf_counter()
        
        if flags[1].is_set():
            break
        flag_wait = False
        for ev in comms:
            if not ev.is_set():
                flag_wait = True
                break
        if flag_wait: continue
        
        flags[2].clear()
        
        # Copy data to local memory
        for i in range(streams_nb):
            local_mem[i][0] = copy(buffer[i][0])
            local_mem[i][1] = buffer[i][1].copy()
        
        for ev in comms:
            ev.clear()
        flags[2].set()
        
        # DEBUG
        t1 = time.perf_counter()
        
        # TODO Process data from buffer
        if mode == 1:
            estim_points = []
            final_points = []
            txt = ""
            for i in range(streams_nb):
                msg = local_mem[i][0]
                idx = int(msg.split('.')[-1])
                img_p = local_mem[i][1].copy()
                if len(img_p) != 6:
                    continue
                img_p_asl = [(0, x, y) for x, y, r in img_p]
                srt_p = find_corresponding_points(img_p_asl)
                est_p = []
                for pi in srt_p:
                    ei = np.dot(H_matrixes[idx], np.append(pi[1:], [1]).transpose())
                    ei = ei / ei[-1]
                    est_p.append(ei[0:2])
                txt += f"[{idx}] : [{len(img_p)}] ,"
                # est_p = np.dot(H_matrixes[idx], np.append(img_p[1:], [1]).transpose())
                # est_p = est_p / est_p[-1]
                # estim_points.append(est_p[0:2])
                estim_points.append(est_p)
            # print(txt)
            # final_points = np.mean(np.array(estim_points), axis=0)
            final_points = np.mean(np.array(estim_points), axis=0)
            # print(final_points)
            for i in range(6):
                print(f"{i+1}: {final_points[i, 0]:7.4f} {final_points[i, 1]:7.4f}, ", end="")
            print("")
        
        # DEBUG
        if mode == 0:
            for pkg in local_mem:
                cv2.imshow(pkg[0], cv2.resize(pkg[1], (480, 360)))
            
            # for i in range(sq):
            #     for j in range(sq):
            #         if i*sq+j >= streams_nb: break
            #         bg[360*i:360*(i+1), 480*j:480*(j+1)] = local_mem[i*sq+j][1][:,:]
            #         cv2.putText(bg, f"[{local_mem[i*sq+j][0]}]", (480*j+20,360*i+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 2)
            # cv2.imshow("images", bg)
            cv2.waitKey(1)
        
        # elif mode == 1:
        #     # for pkg in local_mem:
        #     #     print(f"Server:THREAD::run_processing: From [{pkg[0]}] received: {pkg[1].flatten()}")
            
        #     lim = 1
        #     if cnt:
        #         tmp = []
        #         for pkg in local_mem:
        #             print(np.array(pkg[1]).shape, end=" ")
        #             tmp.append(pkg[1])
        #         print("")
        #         plist.append(tmp)
        #     # if cnt == lim:
        #     #     to_save = np.array(plist)
        #     #     print(to_save.shape)
            
        #     txt = ""
        #     blank = np.zeros((160,640), dtype=np.uint8)
        #     for i, pkg in enumerate(local_mem):
        #         txt = f"From [{pkg[0]}] received: {pkg[1].flatten()}"
        #         cv2.putText(blank, txt, (20,25*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
        #     if cnt == 0:
        #         av_time = t1-t0
        #     else:
        #         av_time = ( cnt*av_time + (t1-t0) ) / (cnt+1)
        #     cnt += 1
        #     txt = f"Elapsed time: {av_time:.6f} [s] | FPS: {1/av_time:.2f}"
        #     cv2.putText(blank, txt, (20,25*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
        #     cv2.imshow("detections", blank)
        #     cv2.waitKey(1)


class Controller():
    
    def __init__(self, config_file: str):
        self.config   = read_config(config_file)
        self.ip_addr  = self.config["CONTROLLER"]["IP"][0]
        self.pub_port = self.config["CONTROLLER"]["PORT"][0]
        self.sender   = ImageSender(f"tcp://{self.ip_addr}:{self.pub_port}", REQ_REP=False)
        self.commands = ["-1", "0", "1", "2", "3"] # TODO Remove hardcoded commands
        
        self.help_msg = """
        "Controller available commands:\n
        "- help          display available commands\n
        "- run  (a)      reset all programs (edge devicies and server)\n
        "        e  (0)  resets edge devices (terminate running process)\n
        "            1   start sending camera frames only\n
        "            2   start sending markers detections only\n
        "            3   start sending both frames and detections\n
        "        s  (0)  reset server\n
        "            1   start receiving images only\n
        "            2   start receiving detectins only\n
        "            3   start receiving both images and detections\n
        "- stop (c)      stops only controller\n
        "        a       stops all running programs and controller\n
        "        e       stops only edge devices\n
        "        s       stops only server\n
        """
        
        self.commands = {
            "help": [],
            "run":  {
                        "a": [],
                        "e": ["0", "1", "2", "3"],
                        "s": ["0", "1", "2", "3"],
                    },
            "stop": {
                        "c": [],
                        "a": [],
                        "e": [],
                        "s": [],
                    }
        }
        
        self.off_e = 8
        self.off_s = 16
    
    def receive_commands(self):
        print("Controller::receive_commands: Availabe commands:")
        print("[-1] Shut down edge devicess and controller\n"
              "[ 0] Shut down edge devices\n"
              "[ 1] Reset edge devices\n"
              "[ 2] Set edge devices to send images only\n"
              "[ 3] Set edge devices to send detections")
        while True:
            req = input("> ")
            if req not in self.commands:
                print(f"Controller::receive_commands::WARNING: Unknown command: [{req}]")
            req = int(req)
            if req == -1:
                # Shut down edge devicess and controller
                print("Controller::receive_commands: Shutting down edge devices and program")
                self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([0], dtype=np.uint8))
                break
            elif req == 0:
                # Shut down edge devices
                print("Controller::receive_commands: Shutting down edge devices")
                self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([0], dtype=np.uint8))
            elif req == 1:
                # Reset edge devices
                print("Controller::receive_commands: Resetting edge devices")
                self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([1], dtype=np.uint8))
            elif req == 2:
                # Set edge devices to send images only
                print("Controller::receive_commands: Enable sending images only")
                self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([2], dtype=np.uint8))
            elif req == 3:
                # Set edge devices to send detections
                print("Controller::receive_commands: Enable sending detections")
                self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([3], dtype=np.uint8))
            else:
                raise ValueError(f"Controller::receive_commands::ERROR: Unknown command: [{req}]")
    
    # def receive_commands(self):
    #     print("Controller::receive_commands: Awaiting command")
    #     off_e = self.off_e
    #     off_s = self.off_s
    #     while True:
    #         req = input("> ")
    #         tokens = req.split(" ")
            
    #         """ Parse input
            
    #             Command codes:
    #             8   reset edge devices
    #             9   start sending images only
    #             10  start sending detectiond only
    #             11  start sending both images and detections
    #             15  stop edge devices
    #             16  reset server
    #             17  start receiving images only
    #             18  start receiving detections only
    #             19  start receiving both images and detecitons
    #             23  stop server
    #         """
    #         if tokens[0] in self.commands.keys():
    #             if tokens[0] == "help":
    #                 print(self.help_msg)
    #             else:
    #                 if len(tokens) == 1 or tokens[1] == "":
    #                     tokens.insert(1, list(self.commands[tokens[0]].keys())[0])
    #                 elif tokens[1] not in self.commands[tokens[0]].keys():
    #                     print(f"Controller::receive_commands: Unknown command: [{tokens[0]} {tokens[1]}]")
    #                     continue
                    
    #                 if tokens[0] == "run":
    #                     if len(tokens) == 2 or tokens[2] == "":
    #                         tokens.insert(2, "0")
    #                     elif tokens[2] not in self.commands[tokens[0]][tokens[1]]:
    #                         print(f"Controller::receive_commands: Unknown command: [{tokens[0]} {tokens[1]} {tokens[2]}]")
    #                         continue
                        
    #                     code = int(tokens[2])
    #                     if tokens[1] == "a":
    #                         self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([code+off_e], dtype=np.uint8))
    #                         time.sleep(0.1)
    #                         self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([code+off_s], dtype=np.uint8))
    #                         print("Controller::receive_commands: Send command to edge devices and server")
    #                     elif tokens[1] == "e":
    #                         self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([code+off_e], dtype=np.uint8))
    #                         print("Controller::receive_commands: Send command to edge devices")
    #                     elif tokens[1] == "s":
    #                         self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([code+off_s], dtype=np.uint8))
    #                         print("Controller::receive_commands: Send command to server")
                            
    #                 elif tokens[0] == "stop":
    #                     if tokens[1] == "c":
    #                         print("Controller::receive_commands: Shut down controller")
    #                         break
    #                     elif tokens[1] == "a":
    #                         self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([15], dtype=np.uint8))
    #                         time.sleep(0.1)
    #                         self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([23], dtype=np.uint8))
    #                         print("Controller::receive_commands: Shut down all devices")
    #                         break
    #                     elif tokens[2] == "e":
    #                         self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([15], dtype=np.uint8))
    #                         print("Controller::receive_commands: Shut down all devices")
    #                     elif tokens[2] == "s":
    #                         self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([23], dtype=np.uint8))
    #                         print("Controller::receive_commands: Shut down server")
    #         else:
    #             print(f"Controller::receive_commands: Unknown command: [{tokens[0]}]")
    #             continue

