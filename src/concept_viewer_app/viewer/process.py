import os

import pandas as pd

from concept_viewer_app.viewer.frex_coherence import get_top_frex_words

ROOT = os.path.abspath(__file__+'/../../../..')
MAX_NB_WORDS = 10

df = None


def process_text(result, wordset, frex_w):
    topic_lists = get_top_frex_words(result, wordset, frex_w, MAX_NB_WORDS)
    topic_lines = ['concept %d: %s' % (i, ' '.join(t_list)) for i, t_list in enumerate(topic_lists)]
    processed_text = '\n'.join(topic_lines)
    return processed_text


def process(min_df, max_df, min_tf, max_tf, frex, dataset):
    global df
    if df is None:
        word_freq_path = os.path.join(ROOT, 'data', dataset, 'df_tf.csv')
        df = pd.read_table(word_freq_path, sep=',')
    filtered_df = df.loc[(df['df_percentile'] >= min_df) & (df['df_percentile'] <= max_df) &
                         (df['tf_percentile'] >= min_tf) & (df['tf_percentile'] <= max_tf)]
    wordset = set(filtered_df['word'])

    def foo(result):
        new_result = result
        new_result.topics = process_text(result, wordset, frex)
        return new_result

    return foo
