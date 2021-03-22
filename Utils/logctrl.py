import os
import time
import datetime
import logging
from logging.handlers import TimedRotatingFileHandler

def CreatePathDirectory(path):
    try:
        path = path.replace("\\", "/")
        p0 = path.rfind('/')
        if p0 != -1 and not os.path.exists(path[:p0]):
            os.makedirs(path[:p0])
            
    except Exception as e:
        print(e)

def GetLogger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    CreatePathDirectory(path)

    rf_handler = logging.handlers.TimedRotatingFileHandler(path, when='midnight', interval=1, backupCount=7,atTime=datetime.time(0, 0, 0, 0))
    formatter  = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s')
    rf_handler.setFormatter(formatter)
    logger.addHandler(rf_handler)

    sh_handler = logging.StreamHandler()
    sh_handler.setFormatter(formatter)
    logger.addHandler(sh_handler)
    
    return logger

def Get_Time_Stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d-%H-%M-%S", local_time)
    return data_head

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val*num
        self.count += num
        self.avg = self.sum / self.count
