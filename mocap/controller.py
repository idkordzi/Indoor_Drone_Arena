import time
import numpy as np

from .imagezmq import ImageSender
from .header import read_config, CMD_OFFSET_EDGE, CMD_OFFSET_SERVER


class Controller():
    
    def __init__(self, config_file: str):
        self.config   = read_config(config_file)
        self.ip_addr  = self.config["CONTROLLER"]["IP"][0]
        self.pub_port = self.config["CONTROLLER"]["PORT"][0]
        self.sender   = ImageSender(f"tcp://{self.ip_addr}:{self.pub_port}", REQ_REP=False)
        
        self.help_msg = "{}".format("Available commands:\n"
                                    "- help           display available commands\n"
                                    "\n"
                                    "- edge    0   reset edge devices\n"
                                    "          1   start sending camera frames only\n"
                                    "          2   start sending markers detections only\n"
                                    "          3   start sending both frames and detections\n"
                                    "\n"
                                    "- server  0   reset server\n"
                                    "          1   start receiving images only\n"
                                    "          2   start receiving detectins only\n"
                                    "          3   start receiving both images and detections\n"
                                    "\n"
                                    "- stop    0   stop all devices\n"
                                    "          1   stop edge devices only\n"
                                    "          2   stop server only\n"
                                    "          3   stop controller only\n")
        self.commands = {
            "help":   [],
            "edge":   ["0", "1", "2", "3"],
            "server": ["0", "1", "2", "3"],
            "stop":   ["0", "1", "2", "3"],
        }
        self.off_e = CMD_OFFSET_EDGE
        self.off_s = CMD_OFFSET_SERVER
    
    def run(self):
        print("Controller::run: Awaiting commands")
        off_e = self.off_e
        off_s = self.off_s
        while True:
            req = input("> ")
            tokens = [tk for tk in req.split(" ") if tk != ""]
            """ Command codes:
                
                8   reset edge devices
                9   start sending images only
                10  start sending detectiond only
                11  start sending both images and detections
                12  ..
                13  ..
                14  ..
                15  stop edge devices
                
                16  reset server
                17  start receiving images only
                18  start receiving detections only
                19  start receiving both images and detecitons
                20  ..
                21  ..
                22  ..
                23  stop server
            """
            if len(tokens) < 1: continue
            if tokens[0] in self.commands.keys():
                if tokens[0] == "help":
                    print(self.help_msg)
                else:
                    if len(tokens) == 1: tokens.append(self.commands[tokens[0]][0])
                    if tokens[1] not in self.commands[tokens[0]]:
                        print(f"Controller::run: Unknown command: [{tokens[0]} {tokens[1]}]")
                        continue
                    code = int(tokens[1])
                    if tokens[0] == "stop":
                        if code == 0:
                            self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([15], dtype=np.uint8))
                            time.sleep(0.1)
                            self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([23], dtype=np.uint8))
                            print("Controller::receive_commands: Stop all devices")
                            break
                        elif code == 1:
                            self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([15], dtype=np.uint8))
                            print("Controller::receive_commands: Stop edge devices")
                        elif code == 2:
                            self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([23], dtype=np.uint8))
                            print("Controller::receive_commands: Stop server")
                        elif code == 3:
                            print("Controller::receive_commands: Stop controller")
                            break
                    elif tokens[0] == "edge":
                        self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([code+off_e], dtype=np.uint8))
                        print(f"Controller::receive_commands: Sending to edge devices: [{code}]")
                    elif tokens[0] == "server":
                        self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([code+off_s], dtype=np.uint8))
                        print(f"Controller::receive_commands: Sending to server: [{code}]")
            else:
                print(f"Controller::run: Unknown command: [{tokens[0]}]")
                continue
    
    # def receive_commands(self):
    #     self.commands = ["-1", "0", "1", "2", "3"]
    #
    #     print("Controller::receive_commands: Availabe commands:")
    #     print("[-1] Shut down edge devicess and controller\n"
    #           "[ 0] Shut down edge devices\n"
    #           "[ 1] Reset edge devices\n"
    #           "[ 2] Set edge devices to send images only\n"
    #           "[ 3] Set edge devices to send detections")
    #     while True:
    #         req = input("> ")
    #         if req not in self.commands:
    #             print(f"Controller::receive_commands::WARNING: Unknown command: [{req}]")
    #         req = int(req)
    #         if req == -1:
    #             # Shut down edge devicess and controller
    #             print("Controller::receive_commands: Shutting down edge devices and program")
    #             self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([0], dtype=np.uint8))
    #             break
    #         elif req == 0:
    #             # Shut down edge devices
    #             print("Controller::receive_commands: Shutting down edge devices")
    #             self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([0], dtype=np.uint8))
    #         elif req == 1:
    #             # Reset edge devices
    #             print("Controller::receive_commands: Resetting edge devices")
    #             self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([1], dtype=np.uint8))
    #         elif req == 2:
    #             # Set edge devices to send images only
    #             print("Controller::receive_commands: Enable sending images only")
    #             self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([2], dtype=np.uint8))
    #         elif req == 3:
    #             # Set edge devices to send detections
    #             print("Controller::receive_commands: Enable sending detections")
    #             self.sender.send_image_pubsub(f"{self.ip_addr}", np.array([3], dtype=np.uint8))
    #         else:
    #             raise ValueError(f"Controller::receive_commands::ERROR: Unknown command: [{req}]")  
