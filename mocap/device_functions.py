import cv2
import numpy as np

from .header import (
    PATTERN,
    PATTERN_PNB,
    move_pattern,
)
from .imagezmq import ImageSender
from .utils import (
    detect_by_markers,
    detect_by_foreground,
    find_corresponding_points,
    get_homography_matrix,
    get_projection_matrix,
    ImageSaver,
)


def new_process_send_images(
    pipe_out,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str,
) -> None:
    
    """ Send images to the server """
    
    cv2.namedWindow("frame")
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    pub_port = config["PUBLISHERS"]["PORT_IMG"][0]
    sender = ImageSender(connect_to=f"tcp://*:{pub_port}", REQ_REP=False)
    print(f"MocapCamera::SUBPROCESS::new_process_send_images: Start processing")
    while True:
        status = pipe_out.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        
        _, frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sender.send_jpg_pubsub(f"{msg}", frame)
    print(f"MocapCamera::SUBPROCESS::new_process_send_images: Stop process")
    
    cv2.destroyWindow("frame")
    
    return


def new_process_send_detections(
    pipe_out,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str,
) -> None:
    
    """ Send detections to the server """

    detection_method = config["DETECTION"].get("METHOD", "MARKERS")
    detection_function = None
    if detection_method == "MARKERS":
        detection_function = lambda frame: detect_by_markers(frame, config["DETECTION"]["MARKERS"])
    elif detection_method == "FOREGROUND":
        cMoG = cv2.createBackgroundSubtractorMOG2()
        detection_function = lambda frame: detect_by_foreground(frame, config["DETECTION"]["FOREGROUND"], cMoG)
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    pub_port = config["PUBLISHERS"]["PORT_PTS"][0]
    sender = ImageSender(connect_to=f"tcp://*:{pub_port}", REQ_REP=False)
    print(f"MocapCamera::SUBPROCESS::new_process_send_detections: Start processing")
    while True:
        status = pipe_out.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        objs = detection_function(frame)
        sender.send_image_pubsub(f"{msg}", np.array(objs, dtype=np.float32))
    print(f"MocapCamera::SUBPROCESS::new_process_send_detections: Stop process")
    return


def new_process_send_all(
    pipe_out,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str,
) -> None:
    
    """ Send both images and detections to the server """
    
    # cv2.namedWindow("detection")
    # cv2.namedWindow("debug")
    
    detection_method = config["DETECTION"].get("METHOD", "MARKERS")
    detection_function = None
    if detection_method == "MARKERS":
        detection_function = lambda frame: detect_by_markers(frame, config["DETECTION"]["MARKERS"])
    elif detection_method == "FOREGROUND":
        # machine = cv2.createBackgroundSubtractorMOG2()
        # machine = cv2.createBackgroundSubtractorKNN()
        background = np.zeros((1080, 1440), dtype=np.uint8)
        foreground_mask = np.ones((1080, 1440), dtype=np.bool8)
        first_iter_flag = True
        detection_function = lambda frame: detect_by_foreground(frame, config["DETECTION"]["FOREGROUND"], 
                                                                machine=None, 
                                                                background=background, 
                                                                foreground_mask=foreground_mask)
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    pub_port_img = config["PUBLISHERS"]["PORT_IMG"][0]
    sender_img = ImageSender(connect_to=f"tcp://*:{pub_port_img}", REQ_REP=False)
    pub_port_det = config["PUBLISHERS"]["PORT_PTS"][0]
    sender_det = ImageSender(connect_to=f"tcp://*:{pub_port_det}", REQ_REP=False)
    print(f"MocapCamera::SUBPROCESS::new_process_send_all: Start processing")
    while True:
        status = pipe_out.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        if first_iter_flag and detection_method == "FOREGROUND":
            background = frame.copy()
            first_iter_flag = False
        objs = detection_function(frame)
        
        i = 1
        x_g = 0
        y_g = 0
        for x, y, a in objs:
            x_g += x * a
            y_g += y * a
            i += a
        x_g = int(x_g / i)
        y_g = int(y_g / i)
        objs = [[x_g, y_g, 0]]
        # blank = np.concatenate([frame[:,:,None], frame[:,:,None], frame[:,:,None]], axis=-1)
        # cv2.circle(blank, (x_g, y_g), int(4), (0,0,255), 6)
        # cv2.imshow("detection", cv2.resize(blank, None, fx=0.5, fy=0.5))
        # cv2.waitKey(1)
        
        _, frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sender_img.send_jpg_pubsub(f"{msg}", frame)
        sender_det.send_image_pubsub(f"{msg}", np.array(objs, dtype=np.float32))
        
    # cv2.destroyWindow("detection")
    # cv2.destroyWindow("debug")
        
    print(f"MocapCamera::SUBPROCESS::new_process_send_all: Stop process")
    return


def new_process_calibrate_camera(
    pipe_out,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str = None,
) -> None:
    
    """ Collect images from camera calibration (offline calibration) """
    
    imgSaver = ImageSaver(
        dir =config["DEVICE"].get("DIR", "device"),
        nb  =config["DEVICE"].get("NB", 0),
        step=config["DEVICE"].get("STEP", 1),
        clr =config["DEVICE"].get("CLEAR", True),
    )
    scale_display = 0.5
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    flag = False
    print(f"MocapCamera::SUBPROCESS::new_process_calibrate_camera: Start processing")
    while True:
        status = pipe_out.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        cv2.imshow("frame", cv2.resize(frame, None, fx=scale_display, fy=scale_display))
        resp = cv2.waitKey(1)
        if resp == ord('q'):
            print(f"MocapCamera::SUBPROCESS::new_process_calibrate_camera: Stop calibrating")
            break
        elif resp == ord('s'):
            flag = imgSaver.save_image(frame)
            print(f"MocapCamera::SUBPROCESS::new_process_calibrate_camera: Frame saved") 
        if flag:
            print(f"MocapCamera::SUBPROCESS::new_process_calibrate_camera: Maximum number of images reached")
            break
    cv2.destroyWindow("frame")
    print(f"MocapCamera::SUBPROCESS::new_process_calibrate_camera: Stop process")
    return


def new_process_calculate_homography_matrix(
    pipe_out,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str=None,
) -> None:
    
    """ Calculate homograpfy matrix using predefined pattern """
    
    save_dir = config["DEVICE"].get("DIR", "device")
    new_pattern = move_pattern(PATTERN, v=[0., 0., 0.])
    scale_display = 0.5
    if msg is None:
        cam_id = "x0"
    else:
        cam_id = msg.split(".")[-1]
    # cam_id = "x0"
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    flag_en = False
    print(f"MocapCamera::SUBPROCESS::new_process_calculate_homography_matrix: Start processing")
    while True:
        status = pipe_out.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        objs = detect_by_markers(frame, config["DETECTION"]["MARKERS"])
        flag_en = (len(objs) == PATTERN_PNB)
        if flag_en:
            indexed_markers = [(0, x, y) for x, y, _ in objs]
            sorted_markers = find_corresponding_points(indexed_markers)
        blank = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        for i in [0,1,2]: blank[:,:,i] = frame
        for x, y, r in objs:
            cv2.circle(blank, (int(x), int(y)), int(r*2), (0,0,255), 2)
        if flag_en:
            for idx, x, y in sorted_markers:
                cv2.putText(blank, f"{idx+1}", (int(x)+10, int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        blank = cv2.resize(blank, None, fx=scale_display, fy=scale_display)
        cv2.putText(blank, f"DET STRUCT SHAPE: {np.array(objs).shape}", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.imshow("frame", blank)
        resp = cv2.waitKey(1)
        if resp == ord('s') and flag_en:
            H = get_homography_matrix(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), objs, new_pattern, True)
            np.save(f"{save_dir}/H_{cam_id}.npy", H)
            print(f"MocapCamera::SUBPROCESS::new_process_calculate_homography_matrix: "
                  f"File saved to [{save_dir}/H_{cam_id}.npy]")
        elif resp == ord('q'):
            break
    cv2.destroyWindow("frame")
    print(f"MocapCamera::SUBPROCESS::new_process_calculate_homography_matrix: Stop process")
    return


def new_process_calculate_projection_matrix(
    pipe_out,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str=None,
) -> None:
    
    save_dir = config["DEVICE"].get("DIR", "device")
    new_pattern = move_pattern(PATTERN, v=[0., 0., 0.])
    scale_display = 0.5
    if msg is None:
        cam_id = "x0"
    else:
        cam_id = msg.split(".")[-1]
    # cam_id = "x0"
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    flag_en = False
    print(f"MocapCamera::SUBPROCESS::new_process_calculate_projection_matrix: Start processing")
    while True:
        status = pipe_out.recv()
        if status == "STOP": break
        with shared_memory.get_lock():
            frame = np.copy(shared_np)
            frame = frame.reshape((im_height, im_width))
        objs = detect_by_markers(frame, config["DETECTION"]["MARKERS"])
        flag_en = (len(objs) == PATTERN_PNB)
        if flag_en:
            indexed_markers = [(0, x, y) for x, y, _ in objs]
            sorted_markers = find_corresponding_points(indexed_markers)
        blank = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        for i in [0,1,2]: blank[:,:,i] = frame
        for x, y, r in objs:
            cv2.circle(blank, (int(x), int(y)), int(r*2), (0,0,255), 2)
        if flag_en:
            for idx, x, y in sorted_markers:
                cv2.putText(blank, f"{idx+1}", (int(x)+10, int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        blank = cv2.resize(blank, None, fx=scale_display, fy=scale_display)
        cv2.putText(blank, f"DET STRUCT SHAPE: {np.array(objs).shape}", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.imshow("frame", blank)
        resp = cv2.waitKey(1)
        if resp == ord('s') and flag_en:
            P = get_projection_matrix(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), objs, new_pattern, True)
            np.save(f"{save_dir}/P_{cam_id}.npy", P)
            print(f"MocapCamera::SUBPROCESS::new_process_calculate_projection_matrix: "
                  f"File saved to [{save_dir}/P_{cam_id}.npy]")
        elif resp == ord('q'):
            break
    cv2.destroyWindow("frame")
    print(f"MocapCamera::SUBPROCESS::new_process_calculate_projection_matrix: Stop process")
    return


def new_process_dummy_process(
    pipe,
    shared_memory,
    im_height: int,
    im_width: int,
    config: dict,
    msg: str=None,
    mode: int=0,
    resize_en: bool=False,
    wait_ns: float=16.0,
) -> None:
    
    print(f"MocapCamera::SUBPROCESS::new_process_dummy_process: Init process")
    cam_id = "x0"
    save_dir = config["DEVICE"].get("DIR", "device")
    pub_port_img = config["PUBLISHERS"]["PORT_IMG"][0]
    pub_port_det = config["PUBLISHERS"]["PORT_PTS"][0]
    cnt = 0
    scale_display = 0.5
    cv2.namedWindow("dummy")
    
    shared_np = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
    
    # Mode 0: Play the role of camera and create frames to process
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
                break
        
        pass

    # Mode 1: Receive frames and do processing
    elif mode == 1:
        
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
            # print(f"MocapCamera::SUBPROCESS::new_process_dummy_process: Frame no. [{cnt}]")
            # cv2.imshow("dummy", frame_copy)
            # resp = cv2.waitKey(1)
            
            # # Display detections on image
            # cnt += 1
            # objs = detect_by_markers(frame, config["DETECTION"]["MARKERS"])
            # blank = np.zeros((im_height, im_width, 3), dtype=np.uint8)
            # for i in [0,1,2]: blank[:,:,i] = frame[:, :]
            # for x, y, r in objs:
            #     cv2.circle(blank, (int(x), int(y)), int(r*2), (0,0,255), 2)
            # blank = cv2.resize(blank, None, fx=scale_display, fy=scale_display)
            # cv2.putText(blank, f"DET STRUCT SHAPE: {np.array(objs).shape}", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # cv2.imshow("dummy", blank)
            # resp = cv2.waitKey(1)
            
            # # Save detections
            # plist = []
            # if resp == ord('p'):
            #     plist.append(objs)
            #     print("MocapCamera::SUBPROCESS::new_process_dummy_process: Points stored")
            # if resp == ord('s'):
            #     to_save = np.array(plist)
            #     np.save(f"{save_dir}/det_{cam_id}.npy", to_save)
            #     print(f"MocapCamera::SUBPROCESS::new_process_dummy_process: "
            #           f"Points saved to [{save_dir}/det_{cam_id}.npy]")
            # if resp == ord('i'):
            #     cv2.imwrite(f"{save_dir}/img_{cam_id}_{cnt:3d}.jpg", frame)
            #     print(f"MocapCamera::SUBPROCESS::new_process_dummy_process: "
            #           f"Frame saved to [{save_dir}/img_{cam_id}_{cnt:3d}.jpg]")
        
        pass
    
    # Mode 2: Receive frames and do processing (and send results)
    elif mode == 2:
        
        # sender = ImageSender(connect_to=f"tcp://*:{pub_port_img}", REQ_REP=False) # send imgs
        # sender = ImageSender(connect_to=f"tcp://*:{pub_port_det}", REQ_REP=False) # send dets
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
            # print(f"MocapCamera::SUBPROCESS::new_process_dummy_process: Frame no. [{cnt}]")
            # _, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # sender.send_jpg_pubsub(f"{msg}", encoded)
            
            # # Send detections
            # objs = detect_by_markers(frame, config["DETECTION"])
            # sender.send_image_pubsub(f"{msg}", np.array(objs, dtype=np.float32))
        
        pass

    cv2.destroyWindow("dummy")
    print(f"MocapCamera::SUBPROCESS::new_process_dummy_process: Stop process")
    return
