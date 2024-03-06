if __name__ == "__main__":
    from mocap.server import Server
    server = Server("configs/config.yaml", 1, False)
    print("MAIN: Start program")
    server.receive_images()
    input("")
    server.stop()
    print("MAIN: Stop program")
