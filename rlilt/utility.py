from setting import *
import numpy as np
from pycommon.settings import *
import torch.nn.functional as func
import kornia.morphology as m
import itertools
import logging
import os
import time
import functools
from collections import defaultdict
from contextlib import contextmanager
import json


def img_transfer(img, x, y):
    img = np.array(img.cpu())
    max_val = max(img.max(), 1)
    x_step = FLIP_SIZE
    y_step = FLIP_SIZE
    consistence = FLIP_CONSISTENCE
    if consistence is True:
        pixel_value = img[y][x]
        for i in range(x - x_step, x + x_step + 1):
            for j in range(y - y_step, y + y_step + 1):
                if 0 <= i < img.shape[1] and 0 <= j < img.shape[0]:
                    img[j][i] = max_val - pixel_value

    else:
        for i in range(x - x_step, x + x_step + 1):
            for j in range(y - y_step, y + y_step + 1):
                if 0 <= i < img.shape[1] and 0 <= j < img.shape[0]:
                    img[j][i] = max_val - img[j][i]

    img = torch.tensor(img > 0.0, dtype=REALTYPE, device=DEVICE)
    return img


def resize_2d_tensor(tensor, height, width, binary=BINARY_IMG):
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    resized_tensor = func.interpolate(tensor, size=(height, width), mode='bicubic', align_corners=False)

    resized_tensor = resized_tensor.squeeze(0).squeeze(0)

    if binary is True:
        resized_tensor[resized_tensor > 0.5] = 1.0
        resized_tensor[resized_tensor <= 0.5] = 0.0
    else:
        resized_tensor[resized_tensor > 1.0] = 1.0
        resized_tensor[resized_tensor <= 0.0] = 0.0

    return resized_tensor


def get_edge(image):

    # print(image.shape)
    image_tensor = image.unsqueeze(0).unsqueeze(0)
    kernel = torch.ones(3, 3, device=DEVICE)
    dilated_image = m.dilation(image_tensor, kernel)
    eroded_image = m.erosion(image_tensor, kernel)
    edge = dilated_image - eroded_image

    wider_kernel = torch.ones(5, 5, device=DEVICE)
    wide_edge = m.dilation(edge, wider_kernel)
    wide_edge[wide_edge > 0.5] = 1.0
    wide_edge[wide_edge <= 0.5] = 0.0

    return wide_edge.squeeze()


class CreateFolder:
    def __init__(self):
        self.path = None
        for idx in itertools.count():
            if os.path.exists(r"./storage/trivial"):
                self.path = os.path.join(r"./storage/trivial", str(idx))
                if not os.path.exists(self.path):
                    os.makedirs(self.path)
                    break
            else:
                print("error : wrong log folder")
                exit()
        print(f"folder created: {self.path}")

    def get_path(self):
        return self.path


def setup_logger(
        name: str,
        folder: str,
        log_filename: str = 'app.log',
        level: int = logging.DEBUG
) -> logging.Logger:

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    if not os.path.exists(folder):
        os.makedirs(folder)

    log_path = os.path.join(folder, log_filename)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')  # 显式指定 utf-8 编码
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger

class CodeProfiler:

    def __init__(self, folder_path):
        self.timings = defaultdict(list)
        self.folder_path = folder_path

    def time_func(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()  # 使用 perf_counter 以获得更高精度
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.timings[func.__name__].append(duration)
            return result

        return wrapper

    @contextmanager
    def time_block(self, name: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.timings[name].append(duration)

    def report(self):

        logger = setup_logger('my_logger', self.folder_path)
        logger.info("\n--- Analysis Report ---")
        if not self.timings:
            logger.info("No timing data was collected")
            return

        grand_total_time = sum(sum(times) for times in self.timings.values())

        if grand_total_time == 0:
            logger.info("The total execution time is zero, so the percentage cannot be calculated.")
            grand_total_time = 1

        report_data = []
        for name, times in self.timings.items():
            total_time = sum(times)
            num_calls = len(times)
            avg_time = total_time / num_calls if num_calls > 0 else 0
            percent_of_total = (total_time / grand_total_time) * 100
            report_data.append({
                "name": name,
                "calls": num_calls,
                "total_time": total_time,
                "avg_time": avg_time,
                "percent_of_total": percent_of_total,
            })

        report_data.sort(key=lambda x: x["total_time"], reverse=True)

        logger.info(f"{'name':<30} | {'calls':>10} | {'total_time (s)':>15} | {'avg_time (s)':>15} | {'percent_of_total (%)':>18}")
        logger.info("-" * 105)

        for data in report_data:
            logger.info(
                f"{data['name']:<30} | {data['calls']:>10} | {data['total_time']:>15.6f} | {data['avg_time']:>15.6f} | {data['percent_of_total']:>17.2f}%")

        logger.info("-" * 105)
        logger.info(f"Total Measurement Time: {grand_total_time:.6f} 秒")
        logger.info("--- End ---\n")


def save_list_json(data, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"The list has been saved to {filename}")
    except Exception as e:
        print(f"Save failed: {e}")

def load_list_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("File does not exist")
        return []
