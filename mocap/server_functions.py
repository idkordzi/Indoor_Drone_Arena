import time
import threading

from copy import copy

import cv2
import numpy as np

from .header import CAMERA_IDS, PATTERN_PNB
from .utils import VideoStreamSubscriber, find_corresponding_points, n_view_traingulation


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
    status = "OK"
    flag_skip = False
    wait_time = 0.01
    start_flag.set()
    time.sleep(wait_time) # so subthreads could start first iteration
    while True:
        if data_read_out.poll():
            status = data_read_out.recv()
            
        # print(f"Server::SUBPROCESS::new_process_start_receiving()::DEBUG: Status [{status}]")
        # time.sleep(0.5)
            
        if status == "STOP":
            print(f"Server::SUBPROCESS::new_process_start_receiving()::INFO: Status [{status}]")
            data_ready_in.send("STOP")
            start_flag.clear()
            kill_flag.set()
            break
        
        if kill_flag.is_set(): # one of the subthreads requested trminating process
            status = "STOP"
            continue
        
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
            status = "IDLE"
            continue
        if status == "OK":
            start_flag.set()
            time.sleep(wait_time)


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
        
        ready_flag.set()
    stream_0.close()
    stream_1.close()
    
    # cv2.destroyWindow("decode")


def new_process_start_processing(
    data_ready_out,
    data_read_in,
    shared_memory: list,
    config: dict,
    receive_mode: int,
    estimation_mode: int=-1,
    initial_pose: np.ndarray=None,
) -> None:
    
    flag_wait = False
    streams_nb = len(shared_memory)
    local_mem = [[[], [], [], []] for _ in range(streams_nb)]
    memory_handle = []
    for i in range(streams_nb):
        frame_np     = np.reshape(np.frombuffer(shared_memory[i][0].get_obj(), dtype=np.uint8), (1080, 1440))
        points_np    = np.frombuffer(shared_memory[i][1].get_obj(), dtype=np.float32)
        points_nb_np = np.frombuffer(shared_memory[i][2].get_obj(), dtype=np.uint8)
        cam_id_np    = np.frombuffer(shared_memory[i][3].get_obj(), dtype=np.uint8)
        memory_handle.append([frame_np, points_np, points_nb_np, cam_id_np])
    
    # Load H matrixes
    if estimation_mode == 1:
        H_matrixes = {}
        for idx in CAMERA_IDS:
            mat = np.load(f"edge/H_{idx}.npy")
            H_matrixes[idx] = mat.copy()
    
    # Load P matrixes
    elif estimation_mode == 2:
        P_matrixes = {}
        P_vector = []
        for idx in CAMERA_IDS:
            mat = np.load(f"edge/P_{idx}.npy")
            P_matrixes[idx] = mat.copy()
            P_vector.append(mat.copy())
    
    
    # DEBUG
    cnt = 0
    av_time = 0.
    # sq = np.ceil(np.sqrt(streams_nb), dtype=int)
    # bg = np.zeros((360*sq, 480*sq), dtype=np.uint8)
    # plist = []
    pose = [0.,0.,0.]
    point_av = False
    
    cv2.namedWindow("cameras")
    cv2.namedWindow("detections")
    
    # pos = np.array([[0,0,0]])
    # fig = plt.figure()
    # ax  = fig.add_subplot(projection="3d")
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
    
    
    # print(f"Server::SUBPROCESS::THREAD::new_process_start_processing(): Start processing")
    
    
    # Main loop
    status = "WAIT"
    data_read_in.send("WAIT")
    while True:
        if data_ready_out.poll():
            status = data_ready_out.recv()
            
        # print(f"Server::SUBPROCESS::new_process_start_processing()::DEBUG: Status [{status}]")
        # time.sleep(0.5)
            
        if status == "STOP":
            print(f"Server::SUBPROCESS::new_process_start_processing()::INFO: Status [{status}]")
            break
        elif status == "WAIT":
            continue
        elif status == "OK":
            for i in range(streams_nb):
                with shared_memory[i][0].get_lock():
                    local_mem[i][0] = np.reshape(memory_handle[i][0].copy(), (1080, 1440))
                
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
            
            
            # DEBUG
            # time.sleep(0.2)
            # status = "WAIT"
            # data_read_in.send("WAIT")
            # continue

            t1 = time.perf_counter()
            
            if receive_mode == 2 or receive_mode == 3:
                
                # Case 1.: Use homography matrixes to estimate pattern position in fixed plane
                if estimation_mode == 0:
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
                elif estimation_mode == 1:
                    
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
                # cv2.imshow(pkg[0], cv2.resize(pkg[1], (480, 360)))
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
                
                txt = ""
                blank = np.zeros((160,1080), dtype=np.uint8)
                for i, pkg in enumerate(local_mem):
                    txt = f"From [{pkg[3]}] received: [{pkg[2]}]"
                    cv2.putText(blank, txt, (20,25*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                # if cnt == 0: av_time = t1-t0
                # else: av_time = ( cnt*av_time + (t1-t0) ) / (cnt+1)
                # cnt += 1
                # txt = f"TIME: RECV: {av_time:.6f} [s] FPS: {1/av_time:.2f} | PROC: {t2-t1:.6f} [s]"
                # cv2.putText(blank, txt, (20,25*(i+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                cv2.imshow("detections", blank)
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
            
            # ...
            
            status = "WAIT"
            data_read_in.send("WAIT")
            continue
    
    # DEBUG
    cv2.destroyWindow("cameras")
    cv2.destroyWindow("detections")
