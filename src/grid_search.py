import copy
import itertools
import json
import os
import random
import threading
import time
import traceback
from queue import Queue

import gc
import numpy
import pandas
import torch

numpy.random.seed(0)


torch.backends.cudnn.benchmark = True
devices = [torch.device("cuda:{}".format(i)) for i in range(torch.cuda.device_count())]

done_ids = set()
queue = Queue()
lock = threading.Lock()

def make_hash(o):
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    other hashable types (including any lists, tuples, sets, and dictionaries).
    """
    if isinstance(o, (set, tuple, list)):
        return tuple([make_hash(e) for e in o])
    elif not isinstance(o, dict):
        return hash(o)
    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)
    return hash(tuple(frozenset(sorted(new_o.items()))))

def init_cuda():
    for device in devices:
        t = torch.tensor([1.]).to(device)
        del t
    torch.cuda.empty_cache()
    gc.collect()

def available_memory(device):
    return torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)

def training_thread(start_gpu):
    global results
    while not queue.empty():
        gpu = start_gpu
        while available_memory(devices[gpu]) < max_mem:
            time.sleep(wait_time)
            gpu = (gpu + 1) % len(devices)
        device = devices[gpu]
        gpu_num = gpus[gpu]
        params = queue.get_nowait()
        params["id"] = abs(make_hash(params))
        if params["id"] in done_ids:
            print("Id {} already exists".format(params["id"]))
            continue
        model = dataset = None
        print("Beginning training run on GPU {}... with id {}".format(gpu_num, params["id"]))
        try:
            os.makedirs(os.path.join(out_dir, "{}/checkpoints/".format(params["id"])), exist_ok=True)
            model = model_cl(params, device)
            dataset = dataset_cl(params, device)
            start = time.perf_counter()
            losses, frame_spec_l1_l2 = train_fn(model, dataset, params, device, os.path.join(out_dir, "{}/checkpoints/".format(params["id"])))
            end = time.perf_counter()
            run_time = end - start
            best_epoch = numpy.argmin(losses)
            best_loss = losses[best_epoch]
            best_l1_l2 = frame_spec_l1_l2[best_epoch]
        except Exception as e:
            if debug:
                print(traceback.format_exc())
            print("WARNING: exception raised while training on GPU {} with params {}".format(gpu_num, params))
            queue.put(params)
        else:
            params["best_loss"] = best_loss
            params["best_epoch"] = best_epoch
            params["run_time"] = run_time
            params["frame_l1"], params["frame_l2"], params["spec_l1"], params["spec_l2"] = best_l1_l2
            with lock:
                results = results.append(params, ignore_index=True)
                results = results.sort_values(by="best_loss")
                results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
            print("Training run complete, results:", params)
        del model
        del dataset
        torch.cuda.empty_cache()
        gc.collect()
        queue.task_done()


def get_done():
    if results is None:
        return done_ids
    for i, row in results.iterrows():
        done_ids.add(row['id'])
    return done_ids


def grid_search(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    global gpus, max_mem, out_dir, wait_time, debug, results, devices,\
        partitions, done_ids

    mode = config["mode"]
    if "gpus" in config.keys():
        gpus = config["gpus"]
        devices = [torch.device("cuda:{}".format(gpu)) for gpu in gpus]
    if "partitions" in config.keys():
        partitions = config["partitions"]
    else:
        partitions = gpus
    max_mem = config["max_mem"]
    max_threads = config["max_threads"]
    out_dir = config["out_dir"]
    shuffle = config["shuffle"]
    debug = config["debug"]
    params = config["params"]
    params = {k: v if isinstance(v, list) else [v] for k, v in params.items()}
    results = pandas.DataFrame(columns=["id", "best_loss", "best_epoch", "run_time"] + list(params.keys()))
    if mode == "product":
        combos = [dict(zip(params.keys(), values)) for values in itertools.product(*params.values())]
    elif mode == "zip":
        max_len = max(len(x) for x in params.values())
        params = {k: v if len(v) > 1 else v * max_len for k, v in params.items()}
        combos = [dict(zip(params.keys(), values)) for values in zip(*params.values())]
    else:
        raise ValueError("Invalid mode \"{}\"".format(mode))
    if shuffle:
        random.shuffle(combos)
    print("Start grid search with %d combos" % len(combos))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(os.path.join(out_dir, "results.csv")):
        results = pandas.read_csv(os.path.join(out_dir, "results.csv"))
    done_ids = get_done()
    if "gs_start" in config:
        combos = combos[config["gs_start"]:]
    for combo in combos:
        queue.put(combo)
    init_cuda()
    for i in range(max_threads):
        time.sleep((i // len(devices)) * wait_time)
        thread = threading.Thread(target=training_thread, args=(i % len(devices),))
        thread.setDaemon(True)
        thread.start()
    queue.join()
    results = results.sort_values(by="best_loss")
    results.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    print("Grid search complete!")
