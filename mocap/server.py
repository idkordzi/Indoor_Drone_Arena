import os
import time
import threading

from copy import copy
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from .header import CAMERA_IDS, PATTERN_PNB, read_config
from .utils import VideoStreamSubscriber, find_corresponding_points, n_view_traingulation


class Server():
    def __init__(self, config_file: str, mode: int=1, save_data: bool=False):
        
        # Read configuration
        self.config = read_config(config_file)
        self.publishers = self.config["PUBLISHERS"]["IP"]
        if mode == 0:
            self.subs_port  = self.config["PUBLISHERS"]["PORT_IMG"][0]
        elif mode == 1:
            self.subs_port  = self.config["PUBLISHERS"]["PORT_PTS"][0]
        self.save_data  = save_data
        self.data_dir   = self.config["SERVER"].get("DIR", "server/")
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
            self.streams.append(VideoStreamSubscriber([p], self.subs_port, self.mode))
            self.comms.append(threading.Event())
            self.buffer.append(["", None])
        
        # Prepare for main thread for processing data
        self.main_thread = None
        
        # Create threading events
        self.flags = []
        for _ in range(3): self.flags.append(threading.Event())

    def run(self) -> None:
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
    
    def listen(self):
        pass
    
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
    recv_mode: int,
    timeout: float
) -> None:
    
    pub_addr = f"{stream.hostnames[0]}:{stream.port}"
    
    # Wait for signal from master
    while True:
        if flags[0].is_set(): break
    print(f"Server::THREAD::run_stream: Start receiving from [{pub_addr}]")
    while True:
        if flags[1].is_set():
            break
        if not flags[2].is_set() or ready.is_set():
            continue
        success, msg, data = stream.receive(timeout=timeout)
        if not success:
            print(f"Server::THREAD::run_stream: Error while reading from [{pub_addr}]")
            break
        buffer[0] = msg
        if recv_mode == 0:
            buffer[1] = cv2.imdecode(np.frombuffer(data, dtype='uint8'), -1)
        elif recv_mode == 1:
            buffer[1] = data.copy()
        ready.set()
    stream.close()


def run_processing(
    buffer: list,
    flags: list,
    comms: list,
    recv_mode: int,
    est_mode: int
) -> None:
    
    flag_wait = False
    streams_nb = len(comms)
    local_mem = [["", None] for _ in range(streams_nb)]
    
    # Load H matrixes
    if est_mode == 0:
        H_matrixes = {}
        for idx in CAMERA_IDS:
            mat = np.load(f"edge/H_{idx}.npy")
            H_matrixes[idx] = mat.copy()
    
    # Load P matrixes
    elif est_mode == 1:
        P_matrixes = {}
        P_vector = []
        for idx in CAMERA_IDS:
            mat = np.load(f"edge/P_{idx}.npy")
            P_matrixes[idx] = mat.copy()
            P_vector.append(mat.copy())
    
    # Wait for signal from master
    while True:
        if flags[0].is_set(): break
    print(f"Server::THREAD::run_processing: Start processing")
    
    # # DEBUG
    # cnt = 0
    # av_time = 0.
    # sq = np.ceil(np.sqrt(streams_nb))
    # bg = np.zeros((360*sq, 480*sq), dtype=np.uint8)
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
        for i in range(streams_nb):
            local_mem[i][0] = copy(buffer[i][0])
            local_mem[i][1] = buffer[i][1].copy()
        for ev in comms:
            ev.clear()
        flags[2].set()
        t1 = time.perf_counter()
        
        if recv_mode == 1:
            
            # Case 1.: Use homography matrixes to estimate pattern position in fixed plane
            if est_mode == 0:
                estim_points = []
                final_points = []
                for i in range(streams_nb):
                    msg = local_mem[i][0]
                    cid = int(msg.split('.')[-1])
                    img_p = local_mem[i][1].copy()
                    if len(img_p) != PATTERN_PNB:
                        continue
                    img_p_asl = [(0, x, y) for x, y, r in img_p]
                    srt_p = find_corresponding_points(img_p_asl)
                    
                    # Case 1.1.: Recive raw points from detectin
                    est_p = []
                    for pi in srt_p:
                        ei = np.dot(H_matrixes[cid], np.append(pi[1:], [1]).transpose())
                        ei = ei / ei[-1]
                        est_p.append(ei[0:2])
                    estim_points.append(est_p)
                        
                    # # Case 1.2.: Recive points after projection
                    # estim_points.append(srt_p)
                    
                final_points = np.mean(np.array(estim_points), axis=0)
                for i in range(PATTERN_PNB):
                    print(f"{i+1}: {final_points[i, 0]:7.4f} {final_points[i, 1]:7.4f}, ", end="")
                print("")
            
            # Case 2.: Use projection matrixes to estimate pattern position in space
            elif est_mode == 1:
                sorted_points = {}
                final_points = []
                for i in range(streams_nb):
                    msg = local_mem[i][0]
                    cid = int(msg.split('.')[-1])
                    img_p = local_mem[i][1].copy()
                    if len(img_p) != PATTERN_PNB:
                        continue
                    img_p_asl = [(0, x, y) for x, y, r in img_p]
                    srt_p = find_corresponding_points(img_p_asl)
                    sorted_points[cid] = srt_p
                # print(f"Server:THREAD::run_processing: Usable packages: {len(sorted_points)}")
                for pi in range(PATTERN_PNB):
                    p_coors = []
                    for ci in sorted_points.keys():
                        p_coors.append(sorted_points[ci][pi][1:3])
                    final_points.append(n_view_traingulation(P_vector, p_coors))
                print(f"PKGS: [{len(sorted_points):1d}] {list(sorted_points.keys())} PTS: [{PATTERN_PNB:1d}] ", end="")
                for i in range(PATTERN_PNB):
                    print(f"{i+1}: {final_points[i][0]:7.3f} {final_points[i][1]:7.3f} {final_points[i][2]:7.3f}, ", end="")
                print("")
            
            pass
        t2 = time.perf_counter()
        # print(f"Server:THREAD::run_processing: RECV: [{t1-t0:.06f}] PROC: [{t2-t1:.6f}] [s]")
        
        
        # DEBUG 
        if recv_mode == 0:
        #     for pkg in local_mem:
        #         cv2.imshow(pkg[0], cv2.resize(pkg[1], (480, 360)))
            
        #     # for i in range(sq):
        #     #     for j in range(sq):
        #     #         if i*sq+j >= streams_nb: break
        #     #         bg[360*i:360*(i+1), 480*j:480*(j+1)] = local_mem[i*sq+j][1][:,:]
        #     #         cv2.putText(bg, f"[{local_mem[i*sq+j][0]}]", (480*j+20,360*i+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 2)
        #     # cv2.imshow("images", bg)
            
        #     cv2.waitKey(1)
            pass
        
        elif recv_mode == 1:
            # for pkg in local_mem:
            #     print(f"Server:THREAD::run_processing: From [{pkg[0]}] received: {pkg[1].flatten()}")

            # if cnt:
            #     tmp = []
            #     for pkg in local_mem:
            #         print(np.array(pkg[1]).shape, end=" ")
            #         tmp.append(pkg[1])
            #     print("")
            #     plist.append(tmp)
            
            # txt = ""
            # blank = np.zeros((160,640), dtype=np.uint8)
            # for i, pkg in enumerate(local_mem):
            #     txt = f"From [{pkg[0]}] received: {pkg[1].flatten()}"
            #     cv2.putText(blank, txt, (20,25*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            # if cnt == 0: av_time = t1-t0
            # else: av_time = ( cnt*av_time + (t1-t0) ) / (cnt+1)
            # cnt += 1
            # txt = f"TIME: RECV: {av_time:.6f} [s] FPS: {1/av_time:.2f} | PROC: {t2-t1:.6f} [s]"
            # cv2.putText(blank, txt, (20,25*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            # cv2.imshow("detections", blank)
            # cv2.waitKey(1)
            pass
