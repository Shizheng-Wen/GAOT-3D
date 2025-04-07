import argparse
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from multiprocessing import Process

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

# Import trainer modules 
from src.trainer.stat import StaticTrainer3D

def setup_logging():
    """
    Set up the logging configuration.
    This function configures logging to use the INFO level and formats log messages with timestamps, log level, and the message.
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S')

class FileParser:
    """
    A utility class for loading configuration files.
    Supports configuration files in TOML and JSON formats.
    """
    def __init__(self, filename: str):
        self.filepath = Path(filename)
        if self.filepath.suffix in [".toml", ".json"]:
            with open(self.filepath, 'r') as f:
                self.kwargs = OmegaConf.load(f)
        else:
            raise NotImplementedError(
                f"File type {self.filepath.suffix} not supported, currently only TOML and JSON are supported."
            )
    
    def add_argument(self, *args, **kwargs):
        """
        Adds a new argument to the configuration.
        If the argument is not already present, it will be added with a default value.
        If an 'action' parameter is provided (e.g., store_true), the default is set to False.
        """
        for arg in args:
            if arg.startswith("--"):
                arg = arg[2:]
            if arg not in self.kwargs:
                # If an action is specified, default the value to False
                if "action" in kwargs:
                    self.kwargs[arg] = False
                else:
                    self.kwargs[arg] = kwargs.get("default", None)
    
    def parse_args(self):
        """
        Convert the stored configuration dictionary into an argparse.Namespace object.
        """
        return argparse.Namespace(**self.kwargs)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    This function supports two modes:
      - Single configuration file mode (using the --config argument)
      - Configuration folder mode (using the --folder argument)
    """
    parser = argparse.ArgumentParser(description="Train Neural Operator Model")
    parser.add_argument("-c", "--config", type=str, help="Path to config file (.toml or .json)")
    parser.add_argument("-f", "--folder", type=str, help="Path to folder containing config files")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (for multiprocessing)")
    parser.add_argument("--num_works_per_device", type=int, default=10, help="Number of jobs to run per device")
    parser.add_argument("--visible_devices", nargs='*', type=int, default=None, help="List of visible GPU device IDs")
    args = parser.parse_args()

    if not args.config and not args.folder:
        parser.error("Please specify either --config or --folder")
    
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.visible_devices))
    
    return args

def collect_config_files(args: argparse.Namespace):
    """
    Collect configuration file paths based on the provided command line arguments.
    Supports both single configuration file mode and configuration folder mode.
    If a folder is provided, it recursively searches for files with .toml or .json extensions.
    """
    config_files = []
    if args.config:
        config_files.append(args.config)
    if args.folder:
        folder = Path(args.folder)
        if folder.is_dir():
            for file in folder.glob("**/*"):
                if file.suffix in [".toml", ".json"]:
                    config_files.append(str(file))
        else:
            logging.error(f"Folder {args.folder} does not exist")
    return config_files

def prepare_paths(arg):
    """
    Manage file paths using the pathlib module.
    Ensures that output directories exist by converting relative paths to absolute ones and creating directories if needed.
    Also initializes a dictionary for recording experimental results.
    """
    basepath = Path(__file__).resolve().parent
    for key in ["ckpt_path", "loss_path", "result_path", "database_path"]:
        path_value = Path(arg.path[key])
        if not path_value.is_absolute():
            abs_path = basepath / path_value
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            arg.path[key] = str(abs_path)
    # Initialize a dictionary to record experimental results and metrics
    arg.datarow = vars(arg).copy()
    arg.datarow.update({
        'nparams': -1,
        'nbytes': -1,
        'p2r edges': -1,
        'r2r edges': -1,
        'r2p edges': -1,
        'training time': np.nan,
        'inference time': np.nan,
        'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'relative error (direct)': np.nan,
        'relative error (auto2)': np.nan,
        'relative error (auto4)': np.nan
    })

def init_distributed_mode(arg):
    """
    Initialize the distributed training environment.
    This function checks environment variables (RANK and WORLD_SIZE) to determine if distributed mode should be enabled.
    If distributed mode is enabled, it initializes the process group using the specified backend.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        arg.setup.rank = int(os.environ['RANK'])
        arg.setup.world_size = int(os.environ['WORLD_SIZE'])
        arg.setup.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        logging.info("Not using distributed mode")
        arg.setup.distributed = False
        arg.setup.rank = 0
        arg.setup.world_size = 1
        arg.setup.local_rank = 0
        return
    
    dist.init_process_group(
        backend=arg.setup.backend,
        init_method='env://',
        world_size=arg.setup.world_size,
        rank=arg.setup.rank
    )
    dist.barrier()

def run_training(arg):
    """
    Execute the training process based on the parsed configuration arguments.
    This function prepares file paths, selects the appropriate Trainer class based on the configuration,
    and runs training and/or testing routines.
    """
    prepare_paths(arg)
    # Select the corresponding Trainer class based on the configuration setting 'trainer_name'
    TrainerDict = {
        "static3d": StaticTrainer3D,
    }
    trainer_name = arg.setup["trainer_name"]
    TrainerClass = TrainerDict.get(trainer_name)
    if TrainerClass is None:
        raise ValueError(f"Unknown trainer: {trainer_name}")
    
    trainer = TrainerClass(arg)
    if arg.setup["train"]:
        if arg.setup["ckpt"]:
            trainer.load_ckpt()
        trainer.fit()
    if arg.setup["test"]:
        trainer.load_ckpt()
        if arg.setup.get("use_variance_test", False):
            trainer.variance_test()
        else:
            trainer.test()
    
    # Record the training results into a CSV database file (only the main process writes the results)
    if getattr(arg.setup, "rank", 0) == 0:
        db_path = arg.path["database_path"]
        if os.path.exists(db_path):
            database = pd.read_csv(db_path)
        else:
            database = pd.DataFrame(columns=arg.datarow.keys())
        database.loc[len(database)] = arg.datarow
        database.to_csv(db_path, index=False)

def run_training_process(arg_file):
    """
    Launch a training task for a single configuration file by calling the current script using subprocess.
    This function constructs a shell command to run the script with the specified configuration file.
    """
    command = f"python {Path(__file__).name} -c {arg_file}"
    logging.info(f"Launching process: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode == 0:
        logging.info(f"Job {arg_file} completed successfully:\n{out.decode('utf-8').strip()}")
    else:
        logging.error(f"Job {arg_file} encountered error:\n{err.decode('utf-8').strip()}")

def run_training_jobs(config_files, debug, num_works_per_device):
    """
    Launch training jobs using different strategies based on the platform and the number of configuration files:
      - Single configuration file: run the training directly.
      - Debug mode: run each configuration sequentially.
      - Windows: use multiprocessing to start separate processes.
      - Linux: group jobs by available devices and run them in batches.
    """
    if len(config_files) == 1:
        parser = FileParser(config_files[0])
        arg = parser.parse_args()
        run_training(arg)
    elif debug:
        for cf in config_files:
            logging.info(f"Running config: {cf}")
            parser = FileParser(cf)
            arg = parser.parse_args()
            run_training(arg)
    elif platform.system() == "Windows":
        processes = []
        for cf in config_files:
            parser = FileParser(cf)
            arg = parser.parse_args()
            p = Process(target=run_training, args=(arg,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    elif platform.system() == "Linux":
        num_devices = torch.cuda.device_count()
        processes = {"cpu": []}
        for i in range(num_devices):
            processes[f"cuda:{i}"] = []
        for cf in config_files:
            parser = FileParser(cf)
            arg = parser.parse_args()
            p = Process(target=run_training_process, args=(cf,))
            if arg.setup.device.startswith("cuda"):
                device_id = int(arg.setup.device[-1])
                processes[f"cuda:{device_id}"].append(p)
            else:
                processes["cpu"].append(p)
        
        max_jobs = max(len(v) for v in processes.values())
        max_runs = (max_jobs + num_works_per_device - 1) // num_works_per_device
        for i in range(max_runs):
            # Start a batch of processes
            for procs in processes.values():
                for p in procs[i * num_works_per_device:(i + 1) * num_works_per_device]:
                    p.start()
            # Wait for the batch of processes to finish
            for procs in processes.values():
                for p in procs[i * num_works_per_device:(i + 1) * num_works_per_device]:
                    p.join()
    else:
        raise NotImplementedError(f"Platform {platform.system()} not supported")

def main():
    setup_logging()
    args = parse_args()
    config_files = collect_config_files(args)
    logging.info(f"Found config files: {config_files}")
    run_training_jobs(config_files, args.debug, args.num_works_per_device)

if __name__ == '__main__':
    main()
