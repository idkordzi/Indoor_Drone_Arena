if __name__ == "__main__":
    from mocap.server import Server
    server = Server("configs/config.yaml")
    print("MAIN: Start program")
    server.listen()
    print("MAIN: Stop program")
