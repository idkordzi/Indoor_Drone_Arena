if __name__ == "__main__":
    from mocap.controller import Controller
    controller = Controller("configs/config.yaml")
    print("MAIN: Start program")
    controller.run()
    print("MAIN: Stop program")
