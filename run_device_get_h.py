if __name__ == "__main__":
    from mocap.edge import MocapCamera
    cam = MocapCamera("configs/config.yaml", True)
    print("MAIN: Start program")
    cam._calibrate_device()
    print("MAIN: Stop program")