if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Runs server to receive detections result from "
                                     "edge devices")
    parser.add_argument("-n", "--network", type=int, choices=[0, 1], default=1,
                        help="Indicates connection to network")
    parser.add_argument("-m", "--mode", type=int, choices=[0, 1, 2, 3], default=0,
                        help="Work mode: 0 - stops server; 1 - receive frames; "
                        "2 - receive detections, 3 - receive both frames and detections")
    parser.add_argument("-l", "--listen", type=int, choices=[0, 1], default=0,
                        help="Runs in listener mode, can be controlled via remote controller, "
                        "overrides other work modes")
    args = parser.parse_args()
    if args.listen == 1:
        print("MAIN::INFO: 'Listener' mode has been enabled")
    
    from mocap.server import Server
    server = Server("configs/config.yaml", args.network == 1)
    print("MAIN: Start program")
    if args.listen == 1:
        server.listen()
    elif args.mode != 0:
        server.run(args.mode)
        input("Press any key to finish\n")
        server.stop()
    print("MAIN: Stop program")
