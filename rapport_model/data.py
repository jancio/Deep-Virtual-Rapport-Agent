#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# PyTorch dataset classes and dataset loaders for rapport model training/testing.
#     Both unimodal and multimodal variants.
#######################################################################################################################


import os
import time
import numpy as np
from tqdm import trange

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from colors import ConsoleColors
from utils import time_diff


#######################################################################################################################
# Unimodal datasets
#######################################################################################################################


class UnimodalBaseDataset(Dataset):
    """
    Base dataset for one modality.

    Features are loaded on-demand.
    Labels are pre-loaded from csv (passed as metadata_labels).
    """

    def __init__(self, metadata_labels, labels_names, features_path, ids, dataset_partition):
        """
        Args:
            metadata_labels (DataFrame): Metadata and labels for each sequence
            labels_names (list): Labels names
            features_path (str): Path to the directory with features
            ids (list): Sequence ids to use for this dataset
            dataset_partition (str): train/val/test
        """

        # Only the specified ids are taken from the dataset file (metadata_labels)
        self.ids = ids
        self.labels_names = labels_names

        self.features_paths = dict()
        self.labels = dict()

        # Create set of ids for faster lookup
        ids_set = set(self.ids)
        for i in trange(len(metadata_labels),
                        desc=f'    loading {ConsoleColors.CC_BLUE}{dataset_partition}{ConsoleColors.CC_END} data'):
            sequence_id = metadata_labels.at[i, 'sequence_id']
            if sequence_id in ids_set:
                self.features_paths[sequence_id] = os.path.join(features_path, sequence_id + '.npy')
                self.labels[sequence_id] = metadata_labels.iloc[i][self.labels_names].tolist()

        # Get postive class weights: calculated as # negative examples / # positive examples
        pos_examples_cnt = np.count_nonzero(list(self.labels.values()), axis=0)
        self.pos_class_weights = torch.Tensor((len(self.labels) - pos_examples_cnt) / pos_examples_cnt)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of a dataset item to load.

        Returns:
            (tuple): Item features and labels
        """
        item_id = self.ids[idx]
        item_features = [torch.Tensor(np.load(self.features_paths[item_id]))]
        item_labels = Variable(torch.Tensor(self.labels[item_id]))

        return item_features, item_labels

    def __len__(self):
        """
        Returns:
            (int): Number of items in the dataset
        """
        return len(self.ids)


def get_unimodal_base_dataset_loaders(metadata_labels, ids, opt):
    """Get dataset loaders for unimodal base dataset.

    Args:
        metadata_labels (DataFrame): Metadata and labels for each sequence
        ids (list): Sequence ids to use for this dataset
        opt: commandline options

    Returns:
        loaders (dict of DataLoader): Dictionary of DataLoaders for desired dataset partitions
    """

    loaders = dict()
    # Which dataset partitions need to be shuffled
    shuffle_flags = {'train': True, 'val': False, 'test': False}

    for dataset_partition in opt.dataset_partitions:
        start_time = time.time()
        dataset = UnimodalBaseDataset(metadata_labels=metadata_labels,
                                      labels_names=opt.labels_names,
                                      features_path=opt.features_path[opt.modality],
                                      ids=ids[dataset_partition],
                                      dataset_partition=dataset_partition)

        loaders[dataset_partition] = DataLoader(dataset=dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=shuffle_flags[dataset_partition],
                                                num_workers=opt.workers_num,
                                                pin_memory=True)

        print(f'\t{dataset_partition} dataset created in {time_diff(start_time)}')

    return loaders


#######################################################################################################################
# Multimodal datasets
#######################################################################################################################


class MultimodalBaseDataset(Dataset):
    """
    Base dataset for multiple modalities.

    Features are loaded on-demand.
    Labels are pre-loaded from csv (passed as metadata_labels).
    """

    def __init__(self, metadata_labels, labels_names, vision_features_path, speech_features_path, ids, dataset_partition):
        """
        Args:
            metadata_labels (DataFrame): Metadata and labels for each sequence
            labels_names (list): Labels names
            vision_features_path (str): Path to the directory with vision features
            speech_features_path (str): Path to the directory with speech features
            ids (list): Sequence ids to use for this dataset
            dataset_partition (str): train/val/test
        """
        # Only the specified ids are taken from the dataset file (metadata_labels)
        self.ids = ids
        self.labels_names = labels_names

        # Avoid 3 repeated iterations over the dataframe
        # self.vision_features_paths = dict([(row['sequence_id'], os.path.join(vision_features_path, row['sequence_id'] + '.npy')) for _, row in metadata_labels.iterrows() if row['sequence_id'] in self.ids])
        # self.speech_features_paths = dict([(row['sequence_id'], os.path.join(speech_features_path, row['sequence_id'] + '.npy')) for _, row in metadata_labels.iterrows() if row['sequence_id'] in self.ids])
        # self.labels = dict([(row['sequence_id'], row[self.labels_names].tolist()) for _, row in metadata_labels.iterrows() if row['sequence_id'] in self.ids])

        self.vision_features_paths = dict()
        self.speech_features_paths = dict()
        self.labels = dict()

        # Create set of ids for faster lookup
        ids_set = set(self.ids)
        for i in trange(len(metadata_labels),
                        desc=f'    loading {ConsoleColors.CC_BLUE}{dataset_partition}{ConsoleColors.CC_END} data'):
            sequence_id = metadata_labels.at[i, 'sequence_id']
            if sequence_id in ids_set:
                self.vision_features_paths[sequence_id] = os.path.join(vision_features_path, sequence_id + '.npy')
                self.speech_features_paths[sequence_id] = os.path.join(speech_features_path, sequence_id + '.npy')
                self.labels[sequence_id] = metadata_labels.iloc[i][self.labels_names].tolist()

        # Get postive class weights: calculated as # negative examples / # positive examples
        pos_examples_cnt = np.count_nonzero(list(self.labels.values()), axis=0)
        self.pos_class_weights = torch.Tensor((len(self.labels) - pos_examples_cnt) / pos_examples_cnt)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of a dataset item to load.

        Returns:
            (tuple): Item features and labels
        """
        item_id = self.ids[idx]

        item_vision = torch.Tensor(np.load(self.vision_features_paths[item_id]))
        item_speech = torch.Tensor(np.load(self.speech_features_paths[item_id]))
        item_features = [item_vision, item_speech]
        item_labels = Variable(torch.Tensor(self.labels[item_id]))
        # print(item_vision.shape, item_speech.shape, item_labels, type(item_labels), type(item_labels[0]), "Check types")

        return item_features, item_labels

    def __len__(self):
        """
        Returns:
            (int): Number of items in the dataset
        """
        return len(self.ids)

def get_multimodal_base_dataset_loaders(metadata_labels, ids, opt):
    """Get dataset loaders for multimodal base dataset.

    Args:
        metadata_labels (DataFrame): Metadata and labels for each sequence
        ids (list): Sequence ids to use for this dataset
        opt: commandline options

    Returns:
        loaders (dict of DataLoader): Dictionary of DataLoaders for desired dataset partitions
    """

    loaders = dict()
    # Which dataset partitions need to be shuffled
    shuffle_flags = {'train': True, 'val': False, 'test': False}

    for dataset_partition in opt.dataset_partitions:
        start_time = time.time()
        dataset = MultimodalBaseDataset(metadata_labels=metadata_labels,
                                        labels_names=opt.labels_names,
                                        vision_features_path=opt.features_path['vision'],
                                        speech_features_path=opt.features_path['speech'],
                                        ids=ids[dataset_partition],
                                        dataset_partition=dataset_partition)

        loaders[dataset_partition] = DataLoader(dataset=dataset,
                                                batch_size=opt.batch_size,
                                                shuffle=shuffle_flags[dataset_partition],
                                                num_workers=opt.workers_num,
                                                pin_memory=True)

        print(f'\t{dataset_partition} dataset created in {time_diff(start_time)}')

    return loaders
