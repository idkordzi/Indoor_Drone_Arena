if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Runs edge device with camera to detect "
                                     "markers/objects")
    parser.add_argument("-n", "--network", type=int, choices=[0, 1], default=1,
                        help="Indicates connection to network")
    parser.add_argument("-c", "--calibrate", type=int, choices=[0, 1, 2, 3], default=0,
                        help="Enables calibration: 0 - disabled; 1 - calibrate camera; "
                        "2 - compute homography matrix; 3 - compute projection matrix")
    parser.add_argument("-d", "--detection", type=int, choices=[0, 1, 2, 3], default=0,
                        help="Enables detection: 0 - disabled; 1 - send frames only; "
                        "2 - send detections only; 3 - send frames and detections")
    # parser.add_argument("-m", "--method", type=int, choices=[0, 1, 2], default=1,
    #                     help="Switches detection method: 0 - disabled; 1 - by markers; "
    #                     "2 - by foreground")
    parser.add_argument("-l", "--listen", type=int, choices=[0, 1], default=0,
                        help="Runs in listener mode, can be controlled via remote controller, "
                        "overrides other work modes")
    args = parser.parse_args()
    if args.listen == 1:
        print("MAIN::INFO: 'Listener' mode has been enabled, all other modes will be disabled")
    elif args.calibrate != 0 and args.detection != 0:
        print("MAIN::WARNING: Both options 'calibrate' and 'detection' were enabled - only "
              "'detection' will be choosen")
    elif args.calibrate == 0 and args.detection == 0:
        args.detection = 2
        print("MAIN::WARNING: No work mode has been chosen - setting to default - detect markers")
    
    from mocap.device import MocapCamera
    edge_dev = MocapCamera("configs/config.yaml", args.network == 1)
    print("MAIN: Start program")
    if args.listen == 1:
        edge_dev.listen()
    elif args.detection == 1:
        edge_dev.run_process_send_images()
        input("Press any key to finish\n")
        edge_dev.stop()
    elif args.detection == 2:
        edge_dev.run_process_send_detections()
        input("Press any key to finish\n")
        edge_dev.stop()
    elif args.detection == 3:
        edge_dev.run_process_send_all()
        input("Press any key to finish\n")
        edge_dev.stop()
    elif args.calibrate == 1:
        edge_dev._run_process_calibrate_camera()
        edge_dev._wait_for_process()
        edge_dev.stop()
    elif args.calibrate == 2:
        edge_dev._run_process_calculate_homography_matrix()
        edge_dev._wait_for_process()
        edge_dev.stop()
    elif args.calibrate == 3:
        edge_dev._run_process_calculate_projection_matrix()
        edge_dev._wait_for_process()
        edge_dev.stop()
    elif args.detection == 0:
        print(f"MAIN::WARNING: No workmode detected - device in idle state")
        input("Press any key to finish\n")
    else:
        print(f"MAIN::ERROR: Unknown error occured, cannot read arguments values")
        exit()
    print("MAIN: Stop program")
