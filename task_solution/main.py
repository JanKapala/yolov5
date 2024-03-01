"""Task solution code"""

import os
from datetime import datetime
from pprint import pprint

import torch.cuda

from detect import run
from task_solution.constants import PREDICTIONS_PATH, DATASET_PATH, LOG_DIR_PATH
from task_solution.custom_logging import init_logging
from task_solution.profiler import PROFILER


def profile() -> None:
    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    devices: list[str | int] = ["cpu"]
    if torch.cuda.is_available():
        devices.append(torch.cuda.current_device())
    else:
        print(
            "CUDA device unavailable! Only cpu profiling will be performed!"
        )

    for device in devices:
        prefix = "with_gpu" if isinstance(device, int) else "only_cpu"
        for file_name in os.listdir(DATASET_PATH):
            init_logging(
                log_file_path=os.path.join(
                    LOG_DIR_PATH,
                    f"performance_log_{prefix}_for_{file_name}_"
                    f"{datetime_string}.txt"
                )
            )
            PROFILER.operation_name_prefix = prefix + "/"
            image_path = os.path.join(DATASET_PATH, file_name)

            run(
                weights="yolov5s.pt",
                device=device,
                source=image_path,
                project=PREDICTIONS_PATH,
                name=f"{file_name.split('.')[0]}_{prefix}",
            )

    print("Aggregated profiling results:")
    pprint(PROFILER.measurements)


if __name__ == "__main__":
    profile()

