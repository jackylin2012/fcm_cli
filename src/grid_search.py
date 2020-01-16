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
from toolbox.helper_functions import get_dataset
import psutil

random.seed(0)

torch.backends.cudnn.benchmark = True

done_ids = set()
queue = Queue()
lock = threading.Lock()


def available_memory(device):
    if device == "cpu" or not torch.cuda.is_available():
        return psutil.virtual_memory()[4]
    else:
        return torch.cuda.get_device_properties(torch.device(device)).total_memory - torch.cuda.memory_allocated(device)


def training_thread(device_idx, ds, config):
    global results
    max_mem = config["max_mem"]
    wait_time = config["wait_time"]
    top_k = config.get("top_k", 10)
    concept_metric = config.get("concept_metric", "dot")

    while not queue.empty():
        available_mem = available_memory(devices[device_idx])
        while available_mem < max_mem:
            print("Not enough memory left ({} < max_mem={}) on device {}. Sleep for {} secs and retry".format(
                available_mem, max_mem, devices[device_idx], wait_time))
            time.sleep(wait_time)
            device_idx = (device_idx + 1) % len(devices)
        device = devices[device_idx]
        dataset_params, fcm_params, fit_params = queue.get_nowait()
        param_id = hashlib.md5(json.dumps({"dataset": dataset_params, "fcm": fcm_params, "fit": fit_params},
                                          sort_keys=True).encode('utf-8')).hexdigest()
        try:
            with lock:
                current_out_dir = os.path.join(out_dir, param_id)
                if os.path.exists(current_out_dir):
                    print("id {} has already been run, skip...".format(param_id))
                    continue
                os.makedirs(current_out_dir)
                params = {**{"dataset." + k: v for k, v in dataset_params.items()},
                          **{"fcm." + k: v for k, v in fcm_params.items()},
                          **{"fit." + k: v for k, v in fit_params.items()}}
                with open(os.path.join(current_out_dir, 'params.json'), 'w') as f:
                    json.dump(params, f, sort_keys=True)
                data_attr = ds.load_data(dataset_params)
            print("Beginning training run on {}... with id {} dataset_params={}, fcm_params={}, fit_params={}".format(
                device, param_id, dataset_params, fcm_params, fit_params))
            print("Save grid search results to {}".format(os.path.abspath(current_out_dir)))
            start = time.perf_counter()
            if torch.cuda.is_available():
                fc_miner = FocusedConceptMiner(current_out_dir, gpu=gpus[device_idx], file_log=True, **fcm_params,
                                               **data_attr)
            else:
                fc_miner = FocusedConceptMiner(current_out_dir, file_log=True, **fcm_params, **data_attr)
            metrics = fc_miner.fit(**fit_params)
            fc_miner.visualize()
            fc_miner.get_concept_words(top_k, concept_metric)
            end = time.perf_counter()
            run_time = end - start
            del fc_miner
            del data_attr
        except Exception as e:
            print(traceback.format_exc())
            print("WARNING: exception raised while training on {} with dataset_params={}, "
                  "fcm_params={}, fit_params={}".format(device, dataset_params, fcm_params, fit_params))
            queue.put((dataset_params, fcm_params, fit_params))
        else:
            best_losses = metrics[:, :-2].min(axis=0)
            best_aucs = metrics[:, -2:].max(axis=0)
            new_result = {**{"dataset." + k: v for k, v in dataset_params.items()},
                          **{"fcm." + k: v for k, v in fcm_params.items()},
                          **{"fit." + k: v for k, v in fit_params.items()},
                          "id": param_id, "run_time": run_time,
                          "total_loss": best_losses[0], "sgns_loss": best_losses[1],
                          "dirichlet_loss": best_losses[2],  "pred_loss": best_losses[3], "div_loss": best_losses[4],
                          "train_auc": best_aucs[0], "test_auc": best_aucs[1]}
            with lock:
                results = results.append(new_result, ignore_index=True)
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
    out_dir = config["out_dir"]
    max_threads = config["max_threads"]
    wait_time = config["wait_time"]
    if torch.cuda.is_available():
        if "gpus" in config.keys():
            gpus = config["gpus"]
        else:
            gpus = list(range(torch.cuda.device_count()))
        devices = ["cuda:{}".format(i) for i in gpus]
    else:
        devices = ["cpu"]
    dataset_params = config["dataset_params"]
    fcm_params = config["fcm_params"]
    fit_params = config["fit_params"]

    results = pandas.DataFrame(columns=["id", "run_time", "total_loss", "sgns_loss", "dirichlet_loss",
                                        "pred_loss", "div_loss", "train_auc", "test_auc"]
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
    results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    print("Grid search complete!")
