"""
convert_data.py
================================
Converts the raw .psv files downloaded in src/data/get_data/download.py into a dataframe useful for analysis.
Further, this converts the binary 0-1 labels to the corresponding utility score.
"""
from definitions import *
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from src.external.evaluate_sepsis_score import compute_prediction_utility
from src.data.dataset import TimeSeriesDataset


def load_to_dataframe():
    """ Convert the raw data to a pandas dataframe. """
    # File locations
    locations = [DATA_DIR + '/raw/' + x for x in ['training_setA', 'training_setB']]

    # Ready to store and concat
    data = []

    # Make dataframe with
    id = 0
    hospital = 1
    for loc in tqdm(locations):
        srt_dir = sorted(os.listdir(loc))
        for file in tqdm(srt_dir):
            id_df = pd.read_csv(loc + '/' + file, sep='|')
            id_df['id'] = id    # Give a unique id
            id_df['hospital'] = hospital    # Note the hospital
            data.append(id_df)
            id += 1
        hospital += 1

    # Concat for df
    df = pd.concat(data)
    df.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)

    # Sort index and reorder columns
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'time'}, inplace=True)
    df = df[['id', 'time'] + [x for x in df.columns if x not in ['id', 'time', 'SepsisLabel']] + ['SepsisLabel']]

    return df


def convert_labels(df):
    """Convert the binary labels to the corresponding utility score.
    The labels are given binary 0-1 values, but are scored according to a pre-defined utility score. Here we convert the
    binary labels onto their corresponding utility value as this value will be more useful for prediction than the
    binary labels.
    """
    def conversion_function(labels):
        if isinstance(labels, pd.DataFrame):
            labels = labels['SepsisLabel']

        # Get same length zeros and ones
        zeros = np.zeros(shape=(len(labels)))
        ones = np.ones(shape=(len(labels)))

        # Get scores for predicting zero or 1
        zeros_pred = compute_prediction_utility(labels.values, zeros, return_all_scores=True)
        ones_pred = compute_prediction_utility(labels.values, ones, return_all_scores=True)

        # Input scores of 0 and 1
        scores = np.concatenate([zeros_pred.reshape(-1, 1), ones_pred.reshape(-1, 1)], axis=1)
        scores = pd.DataFrame(index=labels.index, data=scores, columns=[0, 1])

        # Make an overall utilty score equal to one_score - zero_score which encodes the benefit of the 1 prediction
        scores['utility'] = scores[1] - scores[0]

        return scores

    scores = df.groupby('id').apply(conversion_function)

    return scores


def get_overall_label(df):
    """ Generates a 1 if the patient developed sepsis at any point. """
    return df.groupby('id')['SepsisLabel'].apply(max)


def create_timeseries_dataset(df):
    """ Creates a src.data.dataset.TimeSeriesDataset from the pandas dataframe. """
    # Remove unwanted cols
    df.drop(['time', 'SepsisLabel'], axis=1, inplace=True)
    columns = list(df.drop(['id'], axis=1).columns)

    # Convert df data to tensor form
    tensor_data = []
    ids = df['id'].unique()
    for id in tqdm(ids):
        data = df[df['id'] == id].drop('id', axis=1)
        tensor_data.append(torch.Tensor(data.values.astype(float)))

    # Create dataset
    dataset = TimeSeriesDataset(data=tensor_data, columns=columns)

    return dataset


if __name__ == '__main__':
    # Create a dataframe
    print('Step 1 of 3: Converting to pandas dataframe.')
    df = load_to_dataframe()
    save_pickle(df, DATA_DIR + '/raw/df.pickle')

    # Convert the scores
    print('Step 2 of 3: Converting to labels.')
    overall_labels = torch.Tensor(get_overall_label(df))
    scores = convert_labels(df)

    # Some things to save
    binary_labels = torch.Tensor(df['SepsisLabel'].values)
    utility_scores = torch.Tensor(scores['utility'])

    # Create a time-series dataset
    print('Step 3 of 3: Converting to TimeSeriesDataset format.')
    dataset = create_timeseries_dataset(df)

    # Save the data
    save_pickle(scores, DATA_DIR + '/processed/labels/full_scores.pickle')
    save_pickle(utility_scores, DATA_DIR + '/processed/labels/utility_scores.pickle')
    save_pickle(binary_labels, DATA_DIR + '/processed/labels/binary.pickle')
    save_pickle(overall_labels, DATA_DIR + '/processed/labels/overall_labels.pickle')
    dataset.save(DATA_DIR + '/raw/data.tsd')
