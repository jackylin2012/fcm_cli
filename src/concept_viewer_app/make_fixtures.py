import json
import os
from collections import OrderedDict
import re

ROOT = os.path.abspath(__file__ + '/..')
DATASET_KEYS = ['window_size']
FCM_KEYS = ['embed_dim', 'nnegs', 'nconcepts', 'lam', 'rho', 'eta']


def get_fields(params, dataset, grid_dir, run_id, run_id_path):
    fields = OrderedDict()
    fields['key'] = '%s:%s:%s' % (dataset, grid_dir, run_id)
    fields['dataset'] = dataset
    fields['grid_dir'] = grid_dir
    fields['run_id'] = run_id
    for key in DATASET_KEYS:
        fields[key] = params['dataset'][key]
    for key in FCM_KEYS:
        fields[key] = params['fcm'][key]
    max_episode = -1
    concept_dir = os.path.join(run_id_path, "concept")
    for concept_file in os.listdir(concept_dir):
        episode_match = re.search('epoch(\d+)\.txt', concept_file)
        if episode_match is None:
            continue
        episode = int(episode_match.group(1))
        if episode > max_episode:
            max_episode = episode
    if max_episode == -1:
        return fields
    fields['topics'] = open(os.path.join(concept_dir, "epoch%d.txt" % max_episode)).read()
    return fields


def save_fixtures(grid_search_path):
    dataset = os.path.basename(os.path.dirname(grid_search_path))
    grid_dir = os.path.basename(grid_search_path)
    data_list = []
    for run_id in os.listdir(grid_search_path):
        run_id_path = os.path.join(grid_search_path, run_id)
        if not os.path.exists(os.path.join(run_id_path, "result.json")):
            continue
        with open(os.path.join(run_id_path, "params.json"), "r") as f:
            params = json.load(f)
            data_dict = OrderedDict()
            data_dict['model'] = 'viewer.result'
            data_dict['fields'] = get_fields(params, dataset, grid_dir, run_id, run_id_path)
            data_list.append(data_dict)
    data_file = os.path.abspath(os.path.join(grid_search_path, 'concept_viewer_fixtures.json'))
    with open(data_file, 'w') as f:
        json.dump(data_list, f)
    return data_file
