import time
import threading

from copy import copy

import cv2
import numpy as np

# from functools import partial
# import matplotlib.pyplot as plt
# import matplotlib.animation as anm

from .header import (
    PATTERN,
    PATTERN_PNB,
    move_pattern,
)
from .utils import (
    VideoStreamSubscriber,
    find_corresponding_points,
    get_projection_matrix,
    n_view_traingulation,
)


def new_process_start_receiving(
    data_ready_in,
    data_read_out,
    shared_memory: list,
    config: dict,
    receive_mode: int,
) -> None:
    
    publishers = config["PUBLISHERS"]["IP"]
    if receive_mode == 1:
        subs_ports = [config["PUBLISHERS"]["PORT_IMG"][0]]
    elif receive_mode == 2:
        subs_ports = [config["PUBLISHERS"]["PORT_PTS"][0]]
    elif receive_mode == 3:
        subs_ports = [config["PUBLISHERS"]["PORT_IMG"][0], config["PUBLISHERS"]["PORT_PTS"][0]]
    timeout = config["SERVER"].get("TIMEOUT", 10.0)

    stream_nb = len(publishers)
    threads = []
    
    kill_flag = threading.Event()
    start_flag = threading.Event()
    ready_flags = [threading.Event() for _ in range(stream_nb)]
    
    try:
        for i in range(stream_nb):
            threads.append(threading.Thread(
                target=new_thread_run_stream,
                args=(
                    kill_flag,
                    start_flag,
                    ready_flags[i],
                    shared_memory[i],
                    receive_mode,
                    timeout,
                    publishers[i],
                    subs_ports,
                )
            ))
        for i in range(stream_nb):
            threads[i].start()  
    except:
        print("Server::SUBPROCESS::new_process_start_receiving()::ERROR: "
              "Unknown error occured - stop receiving")
        data_ready_in.send("STOP")
    
    # Main loop
    flag_skip = False
    wait_time = 0.01
    start_flag.set()
    time.sleep(wait_time) # so subthreads could start first iteration
    status = "IDLE"
    while True:
        if data_read_out.poll():
            status = data_read_out.recv()
            
        # print(f"Server::SUBPROCESS::new_process_start_receiving()::DEBUG: Status [{status}]")
        # time.sleep(0.5)
        
        # stop receiving data
        if status == "STOP":
            print(f"Server::SUBPROCESS::new_process_start_receiving()::INFO: Status [{status}]")
            data_ready_in.send("STOP")
            start_flag.clear()
            kill_flag.set()
            break
        
        if kill_flag.is_set(): # one of the subthreads requested terminating the process
            status = "STOP"
            continue
        
        # wait for parallel process to copy data from shared memory
        if status == "WAIT":
            start_flag.clear()
            
            flag_skip = False
            for ev in ready_flags:
                if not ev.is_set(): flag_skip = True
            if flag_skip: continue
            
            # flag_wait = True
            # while flag_wait:
            #     flag_wait = False
            #     for ev in ready_flags:
            #         if not ev.is_set():
            #             flag_wait = True
            
            data_ready_in.send("OK")
            status = "OK"
            continue
        
        if status == "OK":
            start_flag.set()
            time.sleep(wait_time)
            status = "IDLE"


def new_thread_run_stream(
    kill_flag: threading.Event,
    start_flag: threading.Event,
    ready_flag: threading.Event,
    shared_memory: list,
    receive_mode: int,
    timeout: float,
    ip_addr: str,
    ip_ports: list,
) -> None:
    
    class VoidReceviver:
        def receive(self, *args, **kwargs): return True, None, None
        def close(self): return
    
    if receive_mode == 1:
        stream_0 = VideoStreamSubscriber([ip_addr], ip_ports[0], 0)
        stream_1 = VoidReceviver()
    elif receive_mode == 2:
        stream_0 = VoidReceviver()
        stream_1 = VideoStreamSubscriber([ip_addr], ip_ports[0], 1)
    elif receive_mode == 3:
        stream_0 = VideoStreamSubscriber([ip_addr], ip_ports[0], 0)
        stream_1 = VideoStreamSubscriber([ip_addr], ip_ports[1], 1)
    
    shared_frame_np     = np.frombuffer(shared_memory[0].get_obj(), dtype=np.uint8)
    shared_points_np    = np.frombuffer(shared_memory[1].get_obj(), dtype=np.float32)
    shared_points_nb_np = np.frombuffer(shared_memory[2].get_obj(), dtype=np.uint8)
    shared_camera_id_np = np.frombuffer(shared_memory[3].get_obj(), dtype=np.uint8)
    
    pubs_addrs = [f"{ip_addr}:{p}" for p in ip_ports]
    print(f"Server::SUBPROCESS::THREAD::run_stream(): Start receiving from {pubs_addrs}")
    while True:
        
        if kill_flag.is_set():
            break
        if not start_flag.is_set():
            continue
        ready_flag.clear()
        success_0, msg_0, data_0 = stream_0.receive(timeout=timeout)
        success_1, msg_1, data_1 = stream_1.receive(timeout=timeout)
        if not (success_0 and success_1) or (msg_0 is None and msg_1 is None):
            print(f"Server::SUBPROCESS::THREAD::run_stream()::ERROR: Cannot read from [{ip_addr}]")
            print(f"Server::SUBPROCESS::THREAD::run_stream()::INFO: "
                  f"Stream for [{ip_addr}] will kill the process")
            kill_flag.set()
            break
        
        # print(msg_0, success_0, data_0)
        # print(msg_1, success_1, data_1)
        
        if msg_0 is not None:
            with shared_memory[0].get_lock():
                shared_frame_np[:] = cv2.imdecode(np.frombuffer(data_0, dtype=np.uint8), -1).flatten()
                
            # cv2.imshow("decode", np.reshape(shared_frame_np, (1080,1440)))
            # print(f"{shared_frame_np.shape}, {shared_frame_np.dtype}")
            # print(shared_frame_np)
        
        if msg_1 is not None:
            points_pkg = data_1.copy()
            points_pkg_size = points_pkg.shape
            with shared_memory[1].get_lock(), shared_memory[2].get_lock():
                shared_points_np[:points_pkg_size[0]*3] = points_pkg.flatten()
                shared_points_nb_np[0] = np.uint8(points_pkg_size[0])
        
        if msg_0 is None: msg_0 = msg_1
        camera_id = int(msg_0.split(".")[-1])
        with shared_memory[3].get_lock():
            shared_camera_id_np[0] = np.uint8(camera_id)
        
        # print(f"Server::SUBPROCESS::THREAD::run_stream()::INFO: Received fomr [{camera_id}]")
        
        ready_flag.set()
    
    stream_0.close()
    stream_1.close()
    
    # cv2.destroyWindow("decode")


def _match_points(packed_points: dict, P_matrixes: dict) -> dict:
    
    # TODO match points by epipolar lines
    
    # NOW use only first detection (only one point can be detected)
    matched_points = {}
    p_list = []
    for cid in packed_points.keys():
        p_list.append(packed_points[cid][0, 0:2])
    matched_points[1] = p_list
    return matched_points


def _estimate_position_in_plane(matched_points: dict, H_matrices: dict) -> dict:
    raise NotImplementedError()


def _estimate_position_in_space(matched_point: dict, P_vector: list) -> dict:
    
    positioned_points = {}
    
    for pt in matched_point.keys():
        coors = matched_point[pt]
        positioned_points[pt] = n_view_traingulation(P_vector, coors)
    
    return positioned_points


def new_process_start_calibrating(
    data_ready_out,
    data_read_in,
    shared_memory: list,
    config: dict,
    receive_mode: int,
) -> None:

    streams_nb = len(shared_memory)
    local_mem = [[[], [], [], []] for _ in range(streams_nb)]
    memory_handle = []
    for i in range(streams_nb):
        frame_np     = np.reshape(np.frombuffer(shared_memory[i][0].get_obj(), dtype=np.uint8), (1080, 1440))
        points_np    = np.frombuffer(shared_memory[i][1].get_obj(), dtype=np.float32)
        points_nb_np = np.frombuffer(shared_memory[i][2].get_obj(), dtype=np.uint8)
        cam_id_np    = np.frombuffer(shared_memory[i][3].get_obj(), dtype=np.uint8)
        memory_handle.append([frame_np, points_np, points_nb_np, cam_id_np])
    
    # camera_ids = [ip.split(".")[-1] for ip in config["PUBLISHERS"]["IP"]]
    im_width   = config["CAMERA"]["WIDTH"]
    im_height  = config["CAMERA"]["HEIGHT"]
    
    save_dir = config["DEVICE"].get("DIR", "device")
    new_pattern = move_pattern(PATTERN, v=[0., 0., 0.])
    flag_en = False
    off_h = 360
    off_w = 480
    cv2.namedWindow("cameras")
    
    print(f"Server::SUBPROCESS::THREAD::new_process_start_processing(): Start processing")
    
    # Main loop
    status = "WAIT"
    data_read_in.send("WAIT")
    while True:
        if data_ready_out.poll():
            status = data_ready_out.recv()
            
        if status == "STOP":
            print(f"Server::SUBPROCESS::new_process_start_processing()::INFO: Status [{status}]")
            break
        elif status == "WAIT":
            continue
        elif status == "OK":
            for i in range(streams_nb):
                with shared_memory[i][0].get_lock():
                    local_mem[i][0] = np.reshape(memory_handle[i][0].copy(), (im_height, im_width))
                
                with shared_memory[i][1].get_lock(), shared_memory[i][2].get_lock():
                    local_mem[i][2] = memory_handle[i][2][0]
                    off = local_mem[i][2]
                    local_mem[i][1] = np.reshape(memory_handle[i][1][:off*3], (off, 3))
                
                with shared_memory[i][3].get_lock():
                    local_mem[i][3] = memory_handle[i][3][0]
                    
            data_read_in.send("OK")
            status = "WORK"
            continue
        elif status == "WORK":
            
            # HERE
            blank = np.zeros((im_height*2, im_width*2, 3), dtype=np.uint8)
            for idx, pkg in enumerate(local_mem):
                
                ih = (idx // 2)
                iw = (idx % 2)
                
                camera_frame = pkg[0].copy()
                for i in [0, 1, 2]:
                    blank[ih*im_height:(ih+1)*im_height, iw*im_width:(iw+1)*im_width, i] = camera_frame
                cv2.putText(blank, f"CAMERA ID {pkg[3]}", (iw*im_width+20, ih*im_height+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 4)
                
                points_array = pkg[1].copy()
                for pt in points_array:
                    cv2.circle(blank, (int(pt[0]), int(pt[1])), int(5), (0,0,255), 4)
                flag_en = (points_array.shape[0] == PATTERN_PNB)
                if flag_en:
                    indexed_markers = [(0, pt[0], pt[1]) for pt in points_array]
                    sorted_markers = find_corresponding_points(indexed_markers)
                    for idx, x, y in sorted_markers:
                        cv2.putText(blank, f"{idx+1}", (iw*im_width+int(x)+10, ih*im_height+int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 4)
                    cv2.putText(blank, f"All 6 points found", (iw*im_width+20, ih*im_height+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 4)
                else:
                    cv2.putText(blank, f"Cannot find all points (av: {points_array.shape[0]} req: 6)", (iw*im_width+20, ih*im_height+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 4)
                
                blank = cv2.resize(blank, (2*off_w, 2*off_h))
                
            cv2.imshow("cameras", blank)
            resp = cv2.waitKey(1)
            if resp == ord('s') and flag_en:
                for idx, pkg in enumerate(local_mem):
                    objs = [(pt[0], pt[0], 0.) for pt in pkg[1]]
                    flag_en = (len(objs) == PATTERN_PNB)
                    if flag_en:
                        P = get_projection_matrix(cv2.cvtColor(pkg[0].copy(), cv2.COLOR_GRAY2RGB), objs, new_pattern, False)
                        np.save(f"{save_dir}/P_{pkg[3]}.npy", P)
                        print(f"MocapCamera::SUBPROCESS::new_process_calculate_projection_matrix: "
                              f"File saved to [{save_dir}/P_{pkg[3]}.npy]")
                    else:
                        print(f"MocapCamera::SUBPROCESS::new_process_calculate_projection_matrix: "
                              f"For camera [{pkg[3]}] cannot calculate projection matrix - "
                              f"number of detection must be equal to pattern size (found: [{len(objs)}])")
            elif resp == ord('q'):
                cv2.destroyWindow("cameras")
                status = "STOP"
                data_read_in.send("STOP")
                continue
            
            status = "WAIT"
            data_read_in.send("WAIT")
            continue


def new_process_start_processing(
    data_ready_out,
    data_read_in,
    shared_memory: list,
    config: dict,
    receive_mode: int,
    estimation_mode: int=2,
    initial_pose: np.ndarray=None,
) -> None:
    
    streams_nb = len(shared_memory)
    local_mem = [[[], [], [], []] for _ in range(streams_nb)]
    memory_handle = []
    for i in range(streams_nb):
        frame_np     = np.reshape(np.frombuffer(shared_memory[i][0].get_obj(), dtype=np.uint8), (1080, 1440))
        points_np    = np.frombuffer(shared_memory[i][1].get_obj(), dtype=np.float32)
        points_nb_np = np.frombuffer(shared_memory[i][2].get_obj(), dtype=np.uint8)
        cam_id_np    = np.frombuffer(shared_memory[i][3].get_obj(), dtype=np.uint8)
        memory_handle.append([frame_np, points_np, points_nb_np, cam_id_np])
    
    camera_ids = [ip.split(".")[-1] for ip in config["PUBLISHERS"]["IP"]]
    im_width   = config["CAMERA"]["WIDTH"]
    im_height  = config["CAMERA"]["HEIGHT"]
    
    # Load H matrixes
    if estimation_mode == 1:
        H_matrixes = {}
        for idx in camera_ids:
            mat = np.load(f"device/H_{idx}.npy")
            H_matrixes[idx] = mat.copy()
    
    # Load P matrixes
    elif estimation_mode == 2:
        P_matrixes = {}
        P_vector = []
        for idx in camera_ids:
            mat = np.load(f"device/P_{idx}.npy")
            P_matrixes[idx] = mat.copy()
            P_vector.append(mat.copy())
    
    flag_process_frames = receive_mode in [1, 3]
    flag_process_points = receive_mode in [2, 3]
    if estimation_mode == 1:
        estimation_func = lambda x: _estimate_position_in_plane(x, H_matrixes)
    elif estimation_mode == 2:
        estimation_func = lambda x: _estimate_position_in_space(x, P_vector)
    
    if flag_process_frames: cv2.namedWindow("cameras")
    if flag_process_points: cv2.namedWindow("detections")
    
    off_h = 360
    off_w = 480
    
    # DEBUG
    cnt = 0
    av_time = 0.
    pose = [0.,0.,0.]
    point_av = False
    
    
    # Show detected points on 3D plot (untested)
    
    # pos  = np.array([[0,0,0]])
    # fig  = plt.figure()
    # ax   = fig.add_subplot(projection="3d")
    # scat = ax.plot(pos[0,0], pos[0,1], pos[0,2], "ro", ms=16)[0]
    # ax.set(xlim3d=[-10,10], ylim3d=[-10,10], zlim3d=[-10,10])
    
    # def update(frame, pos_a, cnt_a):
    #     pos = pos_a
    #     cnt_a[0] += 1
    #     pos_c = (np.random.rand(1,3) - 0.5).astype(np.float32)
    #     pos += pos_c
    #     cnt = cnt_a[0]
    #     scat.set_data(pos[:, :2].T)
    #     scat.set_3d_properties(pos[:, 2])
    #     fig.suptitle(f"frame: {cnt}")
    #     return scat
    
    # ani = anm.FuncAnimation(fig, partial(update, pos_a=pos, cnt_a=cnt), interval=int(1000/30), cache_frame_data=False)
    # plt.show()
    
    
    print(f"Server::SUBPROCESS::THREAD::new_process_start_processing(): Start processing")
    
    # Main loop
    status = "WAIT"
    data_read_in.send("WAIT")
    while True:
        if data_ready_out.poll():
            status = data_ready_out.recv()
            
        if status == "STOP":
            print(f"Server::SUBPROCESS::new_process_start_processing()::INFO: Status [{status}]")
            break
        elif status == "WAIT":
            continue
        elif status == "OK":
            for i in range(streams_nb):
                with shared_memory[i][0].get_lock():
                    local_mem[i][0] = np.reshape(memory_handle[i][0].copy(), (im_height, im_width))
                
                with shared_memory[i][1].get_lock(), shared_memory[i][2].get_lock():
                    local_mem[i][2] = memory_handle[i][2][0]
                    off = local_mem[i][2]
                    local_mem[i][1] = np.reshape(memory_handle[i][1][:off*3], (off, 3))
                
                with shared_memory[i][3].get_lock():
                    local_mem[i][3] = memory_handle[i][3][0]
                    
            data_read_in.send("OK")
            status = "WORK"
            continue
        elif status == "WORK":
            
            # t1 = time.perf_counter()
            if flag_process_frames:
                pass
            
                blank = np.zeros((off_h*2,off_w*2, 1), dtype=np.uint8)
                for idx, pkg in enumerate(local_mem):
                    ih = (idx // 2)
                    iw = (idx % 2)
                    blank[ih*off_h:(ih+1)*off_h, iw*off_w:(iw+1)*off_w, :] = cv2.resize(pkg[0], (off_w, off_h))
                    cv2.putText(blank, f"CAM ID {pkg[3]}", (iw*off_w+20, ih*off_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.imshow("cameras", blank)
            
            # t2 = time.perf_counter()
            if flag_process_points:
                packed_points = {}
                for i in range(streams_nb):
                    packed_points[local_mem[i][3]] = local_mem[i][1].copy()
                matched_points = _match_points(packed_points, P_matrixes)
                positioned_points = estimation_func(matched_points)
                
                txt = "Estimated pose:\n"
                board = np.zeros((960, 360), dtype=np.uint8)
                if len(positioned_points.keys()) == 0:
                    txt += "No points detected"
                    cv2.putText(board, txt, (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                else:
                    for pt in positioned_points.keys():
                        coors = positioned_points[pt]
                        txt += f"P {pt}: [{coors[0]:0.3f} {coors[0]:0.3f} {coors[0]:0.3f}]\n"
                    cv2.putText(board, txt, (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.namedWindow("detections")
            
            # t3 = time.perf_counter()
            # print(f"Server:THREAD::run_processing: DISP: [{t2-t1:.06f}] PROC: [{t3-t2:.6f}] [s]")
            
            status = "WAIT"
            data_read_in.send("WAIT")
            continue


            # ################################################################
            t1 = time.perf_counter()
            if receive_mode == 2 or receive_mode == 3:
                
                # Case 1.: Use homography matrixes to estimate pattern position in fixed plane
                if estimation_mode == 1:
                    estim_points = []
                    final_points = []
                    for i in range(streams_nb):
                        cid = local_mem[i][3]
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
                elif estimation_mode == 2:
                    
                    break_flag = False
                    
                    sorted_points = {}
                    final_points = []
                    for i in range(streams_nb):
                        
                        cid = local_mem[i][3]
                        img_p = local_mem[i][1].copy()
                        
                        if img_p.shape[0] < 1:
                            break_flag = True
                            break
                        
                        sorted_points[cid] = [(0, img_p[0,0], img_p[0,1])]
                        continue
                        
                        img_p = local_mem[i][1].copy()
                        if len(img_p) != PATTERN_PNB:
                            continue
                        img_p_asl = [(0, x, y) for x, y, r in img_p]
                        srt_p = find_corresponding_points(img_p_asl)
                        sorted_points[cid] = srt_p
                    
                    # print(f"Server:THREAD::run_processing: Usable packages: {len(sorted_points)}")
                    # for pi in range(PATTERN_PNB):
                    #     p_coors = []
                    #     for ci in sorted_points.keys():
                    #         p_coors.append(sorted_points[ci][pi][1:3])
                    #     final_points.append(n_view_traingulation(P_vector, p_coors))
                    
                    if not break_flag:
                    
                        p_coors = []
                        for ci in sorted_points.keys(): 
                            p_coors.append(sorted_points[ci][0][1:3])
                        final_points.append(n_view_traingulation(P_vector, p_coors))
                        # print(f"{0}: {final_points[0][0]:7.3f} {final_points[0][1]:7.3f} {final_points[0][2]:7.3f}, ", end="\n")
                        
                        if len(final_points[0]) != 3:
                            point_av = False
                        else:
                            point_av = True
                            pose[0] = final_points[0][0]
                            pose[1] = final_points[0][1]
                            pose[2] = final_points[0][2]
                    
                    else:
                        # print(f"Server:THREAD::run_processing::WARNING: No points detected")
                        point_av = False
                    
                    # print(f"PKGS: [{len(sorted_points):1d}] {list(sorted_points.keys())} PTS: [{PATTERN_PNB:1d}] ", end="")
                    # for i in range(PATTERN_PNB):
                    #     print(f"{i+1}: {final_points[i][0]:7.3f} {final_points[i][1]:7.3f} {final_points[i][2]:7.3f}, ", end="")
                    # print("")
                
                pass
            
            t2 = time.perf_counter()
            # print(f"Server:THREAD::run_processing: RECV: [{t1-t0:.06f}] PROC: [{t2-t1:.6f}] [s]")
            
            status = "WAIT"
            data_read_in.send("WAIT")
            continue
            
            # DEBUG 
            if receive_mode == 1:
                
                off_h = 360
                off_w = 480
                blank = np.zeros((off_h*2,off_w*2), dtype=np.uint8)
                for idx, pkg in enumerate(local_mem):
                    ih = (idx // 2)
                    iw = (idx % 2)
                    blank[ih*off_h:(ih+1)*off_h, iw*off_w:(iw+1)*off_w] = cv2.resize(pkg[0], (off_w, off_h))
                    cv2.putText(blank, f"CAMERA ID {pkg[3]}", (iw*off_w+20, ih*off_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.imshow("cameras", blank)
                
                # for i in range(sq):
                #     for j in range(sq):
                #         if i*sq+j >= streams_nb: break
                #         bg[360*i:360*(i+1), 480*j:480*(j+1)] = local_mem[i*sq+j][1][:,:]
                #         cv2.putText(bg, f"[{local_mem[i*sq+j][0]}]", (480*j+20,360*i+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 2)
                # cv2.imshow("images", bg)
                
                cv2.waitKey(1)
                pass
            
            elif receive_mode == 2:
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
                # blank = np.zeros((160,1080), dtype=np.uint8)
                # for i, pkg in enumerate(local_mem):
                #     txt = f"From [{pkg[3]}] received: [{pkg[2]}]"
                #     cv2.putText(blank, txt, (20,25*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                # # if cnt == 0: av_time = t1-t0
                # # else: av_time = ( cnt*av_time + (t1-t0) ) / (cnt+1)
                # # cnt += 1
                # # txt = f"TIME: RECV: {av_time:.6f} [s] FPS: {1/av_time:.2f} | PROC: {t2-t1:.6f} [s]"
                # # cv2.putText(blank, txt, (20,25*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                # cv2.imshow("detections", blank)
                # cv2.waitKey(1)
                
                txt = ""
                board = np.zeros((180,1080), dtype=np.uint8)
                for i, pkg in enumerate(local_mem):
                    txt = f"From [{pkg[3]}] received: [{pkg[2]}]"
                    cv2.putText(board, txt, (20,25*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                if cnt == 0: av_time = t2-t1
                else: av_time = (cnt*av_time + (t2-t1)) / (cnt+1)
                cnt += 1
                txt = f"TIME: TOTAL: {av_time:.6f} [s] FPS: {1/av_time:.2f} | PROC: {t2-t1:.6f} [s]"
                cv2.putText(board, txt, (20,25*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                if point_av:
                    txt = f"Estimated pose: [{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f}]"
                else:
                    txt = "No points detected"
                cv2.putText(board, txt, (20,25*(i+3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                cv2.imshow("detections", board)
                cv2.waitKey(1)
                pass
            
            elif receive_mode == 3:
                
                # pos_c = (np.random.rand(1,3) - 0.5).astype(np.float32)
                # pose += pos_c
                
                off_h = 360
                off_w = 480
                blank = np.zeros((off_h*2,off_w*2, 3), dtype=np.uint8)
                for idx, pkg in enumerate(local_mem):
                    ih = (idx // 2)
                    iw = (idx % 2)
                    for i in [0,1,2]: blank[ih*off_h:(ih+1)*off_h, iw*off_w:(iw+1)*off_w, i] = cv2.resize(pkg[0], (off_w, off_h))
                    pt_len = pkg[2]
                    for i in range(pt_len):
                        x = pkg[1][i, 0]
                        y = pkg[1][i, 1]
                        
                        x = int(x / 1440 * off_w) + iw*off_w
                        y = int(y / 1080 * off_h) + ih*off_h
                        
                        cv2.circle(blank, (int(x), int(y)), int(4), (0,0,255), 2)
                    cv2.putText(blank, f"CAM ID {pkg[3]}", (iw*off_w+20, ih*off_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(blank, f"CNT: {cnt}", (off_w-40, off_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.imshow("cameras", blank)
                
                txt = ""
                board = np.zeros((180,1080), dtype=np.uint8)
                for i, pkg in enumerate(local_mem):
                    txt = f"From [{pkg[3]}] received: [{pkg[2]}]"
                    cv2.putText(board, txt, (20,25*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                if cnt == 0: av_time = t2-t1
                else: av_time = (cnt*av_time + (t2-t1)) / (cnt+1)
                cnt += 1
                txt = f"TIME: TOTAL: {av_time:.6f} [s] FPS: {1/av_time:.2f} | PROC: {t2-t1:.6f} [s]"
                cv2.putText(board, txt, (20,25*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                if point_av:
                    txt = f"Estimated pose: [{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f}]"
                else:
                    txt = "No points detected"
                cv2.putText(board, txt, (20,25*(i+3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                cv2.imshow("detections", board)
                cv2.waitKey(1)
                pass
            
            status = "WAIT"
            data_read_in.send("WAIT")
            continue
        
        # ################################################################
    
    if flag_process_frames: cv2.destroyWindow("cameras")
    if flag_process_points: cv2.destroyWindow("detections")

