if __name__ == "__main__":
    from mocap.edge import MocapCamera
    edge = MocapCamera("configs/config.yaml", True)
    print("MAIN: Start program")
    edge.listen()
    print("MAIN: Stop program")
