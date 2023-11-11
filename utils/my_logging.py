import logging
import time
import os
import torch
import requests

import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, net,config_file, order,mode="train"):
        parent_path = f"work/{net}/{config_file[13:]}/{str(order)}"
        model_path = f"work/{net}/{config_file[13:]}/{str(order)}"
        self.config_file = config_file
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        filename = self.get_time()
        log_file = os.path.join(parent_path, f"{filename}.{mode}.log")
        self.loss_file = os.path.join(parent_path, f"{filename}.{mode}.jpg")
        self.model_file = os.path.join(model_path, f"{filename}.pkl")
        self.acc_file = os.path.join(parent_path, f"{filename}.{mode}.acc.jpg")
        logging.basicConfig(level=logging.INFO, filename=log_file)#log_file
        self.logger = logging.getLogger(mode)
        if os.path.exists("send_key.txt"):
            with open("send_key.txt", "r") as f:
                self.send_key = f.readline()
        else:
            self.send_key = None

    def get_time(self):
        return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())

    def info(self, message):
        self.logger.info(f"{self.get_time()}:{message}")

    def log_finish(self, best_iou):
        current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        msg = f"Training process finished at {current_time} with best IOU {best_iou}"
        self.logger.info(msg)
        # self.push(title=f"Task:{self.config_file} ok!", msg=msg)

    def plot_loss(self, epoch_list, loss_list):
        plt.plot(epoch_list, loss_list)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(self.loss_file)

    def plot_acc(self, epoch_list, mIoU_list, mF1_list):
        plt.plot(epoch_list, mIoU_list, label="mIoU")
        plt.plot(epoch_list, mF1_list, label="mF1")
        plt.xlabel("steps")
        plt.savefig(self.acc_file)

    def save_model(self, model):
        torch.save(
            model.state_dict(), self.model_file, _use_new_zipfile_serialization=False
        )

    def push(self, title, msg):
        if self.send_key:
            requests.post(
                url=f"https://sctapi.ftqq.com/{self.send_key}.send",
                data={"text": title, "desp": msg},
            )
