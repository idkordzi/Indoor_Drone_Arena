if __name__ == "__main__":
    from mocap.server import Controller
    controller = Controller("configs/config.yaml")
    print("MAIN: Start program")
    controller.receive_commands()
