if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Runs remote controller to communicate with "
                                     "edge devices and server, requires other devices to be "
                                     "run in 'listener' mode")
    parser.add_argument("-n", "--network", type=int, choices=[0, 1], default=1,
                        help="Indicates connection to network")
    args = parser.parse_args()
    
    from mocap.controller import Controller
    controller = Controller("configs/config.yaml", args.network == 1)
    print("MAIN: Start program")
    controller.run()
    print("MAIN: Stop program")
