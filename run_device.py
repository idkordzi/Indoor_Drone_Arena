if __name__ == "__main__":
    from mocap.edge import MocapCamera
    edge = MocapCamera("configs/config.yaml", True)
    print("MAIN: Start program")
    edge.run_detection_and_sending()
    input("")
    edge.stop()
    print("MAIN: Stop program")
