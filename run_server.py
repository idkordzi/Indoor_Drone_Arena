if __name__ == "__main__":
    from mocap.server import Server
    server = Server("configs/config.yaml", 1)
    print("MAIN: Start program")
    server.run()
    input("")
    server.stop()
    print("MAIN: Stop program")
