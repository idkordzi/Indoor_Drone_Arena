from .imagezmq import ImageSender
from .header import read_config


class Controller():
    
    def __init__(self, config_file: str, network_connected: bool=True):
        self.config   = read_config(config_file)
        self.ip_addr  = self.config["CONTROLLER"]["IP"][0]
        self.pub_port = self.config["CONTROLLER"]["PORT"][0]
        self.sender   = ImageSender(f"tcp://{self.ip_addr}:{self.pub_port}", REQ_REP=False)

        raise NotImplementedError()
    
    def run(self):
        raise NotImplementedError()
