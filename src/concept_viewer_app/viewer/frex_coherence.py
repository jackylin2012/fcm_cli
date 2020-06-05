import json
import os
import re

import numpy as np
import scipy
import scipy.stats
import torch
from scipy.special import logsumexp

from toolbox.helper_functions import get_dataset

ROOT = os.path.abspath(__file__+'/../../../..')
DECODER_CACHE = {}


def load_decoder(run_id_path, dataset):
    dataset_class = get_dataset(dataset)
    ds = dataset_class()
    with open(os.path.join(run_id_path, "params.json"), "r") as f:
        params = json.load(f)
    data_filename = ds.get_data_filename(params["dataset"])
    if data_filename in DECODER_CACHE:
        return DECODER_CACHE[data_filename]
    data = ds.load_data(params["dataset"])
    decoder = data["vocab"]
    DECODER_CACHE[data_filename] = decoder
    return decoder


def load_dotproduct_distance(run_id_path):
    max_episode_file = None
    max_episode = -1
    model_dir = os.path.join(run_id_path, "model")
    for model_file in os.listdir(model_dir):
        episode_match = re.search('epoch(\d+)\.pytorch', model_file)
        if episode_match is None:
            continue
        episode = int(episode_match.group(1))
        if episode > max_episode:
            max_episode = episode
            max_episode_file = model_file
    if max_episode_file is None:
        return None
    state = torch.load(os.path.join(model_dir, max_episode_file), map_location=torch.device('cpu'))
    topic_vectors = state['embedding_t'].cpu().numpy()
    word_vectors = state['embedding_i.weight'].cpu().numpy().T
    return -np.dot(topic_vectors, word_vectors)


def distance_to_distribution(distance, function=lambda x: 1 / x, normalize=True):
    """
    Input: T * V array. 
    """
    distribution = function(distance)
    if normalize:
        distribution = distribution / distribution.sum(axis=1, keepdims=True)
    return distribution


def ecdf(arr):
    """Calculate the empirical CDF values for all elements in a 1D array."""
    return scipy.stats.rankdata(arr, method='max') / arr.size


def calculate_frex(run_id_path, topics, w):
    """Calculate FREX for all words in a topic model.

    See R STM package for details:
    https://cran.r-project.org/web/packages/stm/vignettes/stmVignette.pdf

    """
    log_beta = np.log(topics)
    log_exclusivity = log_beta - logsumexp(log_beta, axis=0)
    exclusivity_ecdf = np.apply_along_axis(ecdf, 1, log_exclusivity)
    freq_ecdf = np.apply_along_axis(ecdf, 1, log_beta)
    np.save(os.path.join(run_id_path, "exclusivity_ecdf.npy"), exclusivity_ecdf)
    np.save(os.path.join(run_id_path, "freq_ecdf.npy"), freq_ecdf)
    return frex_score(exclusivity_ecdf, freq_ecdf, w)


def frex_score(exclusivity_ecdf, freq_ecdf, w):
    return 1. / (w / exclusivity_ecdf + (1 - w) / freq_ecdf)


def get_topics(decoder, frex, wordset, topn):
    topics = []
    for word_indices in (-frex).argsort(axis=1):
        words = []
        for i in word_indices:
            if decoder[i] in wordset:
                words.append(decoder[i])
            if len(words) >= topn:
                break
        topics.append(words)
    return topics


def get_top_frex_words(result, wordset, frex_w, topn):
    run_id_path = os.path.join(ROOT, "grid_search", result.dataset, result.grid_dir, result.run_id)
    decoder = load_decoder(run_id_path, result.dataset)
    excl_file = os.path.join(run_id_path, "exclusivity_ecdf.npy")
    freq_file = os.path.join(run_id_path, "freq_ecdf.npy")
    if os.path.exists(excl_file) and os.path.exists(freq_file):
        exclusivity_ecdf = np.load(excl_file)
        freq_ecdf = np.load(freq_file)
        frex = frex_score(exclusivity_ecdf, freq_ecdf, frex_w)
    else:
        distance = load_dotproduct_distance(run_id_path)
        if distance is None:
            return []
        distribution = distance_to_distribution(distance, function=lambda x: 1 / x, normalize=False)
        frex = calculate_frex(run_id_path, distribution, w=frex_w)
    topics = get_topics(decoder, frex, wordset, topn)
    return topics

