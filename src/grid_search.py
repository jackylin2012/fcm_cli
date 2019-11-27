import hashlib
import itertools
import json
import os
import random
import threading
import time
import traceback
from queue import Queue

import gc
import pandas
import torch

from fcm import FocusedConceptMiner
from util.helper_functions import get_dataset

random.seed(0)

torch.backends.cudnn.benchmark = True

done_ids = set()
queue = Queue()
lock = threading.Lock()


def available_memory(device):
    return torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)


def training_thread(start_gpu, ds, config):
    global results
    max_mem = config["max_mem"]
    wait_time = config["wait_time"]
    top_k = config["top_k"]
    concept_metric = config["concept_metric"]

    while not queue.empty():
        gpu = start_gpu
        while available_memory(devices[gpu]) < max_mem:
            time.sleep(wait_time)
            gpu = (gpu + 1) % len(devices)
        gpu_num = gpus[gpu]
        dataset_params, fcm_params, fit_params = queue.get_nowait()
        param_id = hashlib.md5(json.dumps({"dataset": dataset_params, "fcm": fcm_params}, sort_keys=True)).hexdigest()
        print("Beginning training run on GPU {}... with id {} dataset_params={}, fcm_params={}, fit_params={}".format(
            gpu_num, param_id, dataset_params, fcm_params, fit_params))
        try:
            with lock:
                output_path = os.path.join(out_dir, param_id)
                if os.path.exists(output_path):
                    print("id {} has already been run, skip...")
                    continue
                os.makedirs(output_path)
                data_attr = ds.load_data(dataset_params)
            print("Save grid search results to {}".format(os.path.abspath(output_path)))
            start = time.perf_counter()
            fc_miner = FocusedConceptMiner(out_dir, gpu=gpu, **fcm_params, **data_attr)
            metrics = fc_miner.fit(**fit_params)
            fc_miner.visualize()
            fc_miner.get_concept_words(top_k, concept_metric)
            end = time.perf_counter()
            run_time = end - start
            del fc_miner
            del data_attr
        except Exception as e:
            print(traceback.format_exc())
            print("WARNING: exception raised while training on GPU {} with dataset_params={}, "
                  "fcm_params={}, fit_params={}".format(gpu_num, dataset_params, fcm_params, fit_params))
            queue.put((dataset_params, fcm_params, fit_params))
        else:
            best_losses = metrics[:, :-2].min(axis=0)
            best_aucs = metrics[:, -2:].max(axis=0)
            new_result = {**{"dataset." + k: v for k, v in dataset_params.items()},
                          **{"fcm." + k: v for k, v in fcm_params.items()},
                          **{"fit." + k: v for k, v in fit_params.items()},
                          "id": param_id, "run_time": run_time,
                          "total_loss": best_losses[0], "sgns_loss": best_losses[1],
                          "diversity_loss": best_losses[2], "pred_loss": best_losses[3],
                          "train_auc": best_aucs[0], "test_auc": best_aucs[1]}
            with lock:
                results = results.append(new_result, ignore_index=True)
                results = results.sort_values(by="best_loss")
                results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
            print("Training run complete, results:", new_result)
        torch.cuda.empty_cache()
        gc.collect()
        queue.task_done()


def grid_search(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    global gpus, out_dir, results, devices

    mode = config["mode"]
    if "gpus" in config.keys():
        gpus = config["gpus"]
    else:
        gpus = list(range(torch.cuda.device_count()))
    out_dir = config["out_dir"]
    max_threads = config["max_threads"]
    wait_time = config["wait_time"]
    devices = [torch.device("cuda:{}".format(i)) for i in gpus]
    dataset_params = config["dataset_params"]
    fcm_params = config["fcm_params"]
    fit_params = config["fit_params"]

    results = pandas.DataFrame(columns=["id", "run_time", "total_loss", "sgns_loss", "diversity_loss",
                                        "pred_loss", "diversity_loss", "train_auc", "test_auc"]
                                       + ["dataset." + k for k in dataset_params.keys()]
                                       + ["fcm." + k for k in fcm_params.keys()]
                                       + ["fit." + k for k in fit_params.keys()])
    dataset_params_len = len(dataset_params.keys())
    fcm_params_len = len(fcm_params.keys())
    if mode == "product":
        # in "product" mode, generate every possible combinations of all possible dataset_params and fcm_params
        combos = [(dict(zip(dataset_params.keys(), values[:dataset_params_len])),
                   dict(zip(fcm_params.keys(), values[dataset_params_len:dataset_params_len + fcm_params_len])),
                   dict(zip(fit_params.keys(), values[dataset_params_len + fcm_params_len:])))
                  for values in itertools.product(*dataset_params.values(), *fcm_params.values(), *fit_params.values())]
    elif mode == "zip":
        # in "zip" mode, zip all dataset_params and fcm_params at same positions
        combos = [(dict(zip(dataset_params.keys(), values[:dataset_params_len])),
                   dict(zip(fcm_params.keys(), values[dataset_params_len:dataset_params_len + fcm_params_len])),
                   dict(zip(fit_params.keys(), values[dataset_params_len + fcm_params_len:])))
                  for values in zip((*dataset_params.values(), *fcm_params.values(), *fit_params.values()))]
    else:
        raise ValueError("Invalid mode \"{}\"".format(mode))
    random.shuffle(combos)
    print("Start grid search with %d combos" % len(combos))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(os.path.join(out_dir, "results.csv")):
        results = pandas.read_csv(os.path.join(out_dir, "results.csv"))
    for combo in combos:
        queue.put(combo)
    dataset_class = get_dataset(config["dataset"])
    ds = dataset_class()
    for i in range(max_threads):
        time.sleep((i // len(devices)) * wait_time)
        thread = threading.Thread(target=training_thread, args=(i % len(devices), ds, config))
        thread.setDaemon(True)
        thread.start()
    queue.join()
    results = results.sort_values(by="best_loss")
    results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    print("Grid search complete!")
