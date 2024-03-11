if __name__ == "__main__":
    from mocap.edge import MocapCamera
    edge = MocapCamera("configs/config.yaml", True)
    print("MAIN: Start program")
    edge._compute_homography_matrix()
    # edge._compute_projection_matrix()
    print("MAIN: Stop program")