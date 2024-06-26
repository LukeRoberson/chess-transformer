'''
The class for managing the dataset
'''

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformer_blocks import GPTConfig

import os
import re
import json
from tqdm import tqdm

from typing import Tuple


class DataSet():
    '''
    The class for loading and managing the dataset

    Methods:
        load:
            Load the dataset from JSON files

        split:
            Split the dataset into training and testing sets

        create_dataloaders:
            Create dataloaders for the training and testing sets

        get_batch:
            Get a batch of data from the training or testing set
    '''

    def __init__(
        self,
        config: GPTConfig,
        dataset_dir: str = './dataset'
    ) -> None:
        '''
        Constructor

        Args:
            config: GPTConfig
                The configuration for the GPT model

            dataset_dir: str
                The directory containing the dataset
        '''

        # Required initial values
        self.config = config
        self.dataset_dir = dataset_dir

        # Important values to be set later
        self.padded_game_list = None
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.train_dataloader = None
        self.test_dataloader = None

    def load(
        self
    ) -> None:
        '''
        Load the dataset from JSON files

        We need to track the longest game so we can later pad shorter games,
            so they're all the same length.
            This is needed to convert to a tensor later on.
        '''

        # A starting point for the game list
        game_list = []

        # Track the longest game
        max_length = 0

        # Get a list of all JSON files in the dataset directory
        for file in tqdm(
            os.listdir(self.dataset_dir),
            desc='Processing JSON files'
        ):
            if file.endswith('.json'):
                try:
                    # Read the JSON file
                    with open(f"{self.dataset_dir}/{file}", "r") as f:
                        moves = json.load(f)

                    # Get the year of the game (only one per file)
                    year = list(moves.keys())[0]

                    # Parse the JSON file for games (by month and game)
                    for month in moves[year]:
                        for game in moves[year][month]:
                            # Remove the move numbers from the PGN
                            game = re.sub(
                                r"\d{1,3}\. ", "",
                                game['pgn']
                            ).strip()

                            # Add the game to the list
                            game_list.append(game)

                            # Find the maximum length of a game
                            if len(
                                self.config.tokenizer.tokenize(game)
                            ) > max_length:
                                max_length = len(
                                    self.config.tokenizer.tokenize(game)
                                )

                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error processing file {file}: {e}")

        # Tokenize and pad each game list to the maximum length
        #   This means all lists are the same size
        #   This is needed to convert to tensor later
        self.padded_game_list = [
            self.config.tokenizer.tokenize(game, pad=True, pad_size=max_length)
            for game in game_list
        ]

    def split(
        self,
        test_size: float = 0.2,
    ) -> None:
        '''
        Split the dataset into training and testing sets

        (1) Creates inputs and targets for the model
            The targets (ground truth) are the inputs shifted right
            by one token
            This is because the model predicts the next token
        (2) Splits each dataset into training and testing sets

        Args:
            test_size: float
                The percentage size of the test set
                The remainder is the training set
        '''

        # Get input and target sequences
        #   Target is input shifted right by one token
        input_sequences = []
        target_sequences = []

        for game in self.padded_game_list:
            input_sequences.append(game[:-1])
            target_sequences.append(game[1:])

        # Split input_sequences and target_sequences into train and test sets
        (
            self.train_data, self.test_data,
            self.train_labels, self.test_labels
        ) = train_test_split(
            input_sequences,
            target_sequences,
            test_size=test_size,
        )

    def create_dataloaders(
        self,
        shuffle: bool = True
    ) -> None:
        '''
        Create dataloaders for the training and testing sets

        (1) Create TensorDatasets for the training and testing sets
        (2) Create DataLoader objects for the training and testing sets

        Args:
            shuffle: bool
                Whether to shuffle the train dataset or not
                Testing dataset is never shuffled
        '''

        # Create TensorDatasets for the training and testing sets
        train_dataset = TensorDataset(
            torch.tensor(self.train_data),
            torch.tensor(self.train_labels)
        )
        test_dataset = TensorDataset(
            torch.tensor(self.test_data),
            torch.tensor(self.test_labels)
        )

        # Create DataLoader objects for the training and testing sets
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

    def get_batch(
        self,
        split: str = 'train'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Get a batch of data from the training or testing set
        This returns a batch of input and target tensors
            They are automatically moved to the right device

        Args:
            split: str
                The split to get the batch from
                Either 'train' or 'test'

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                The input and target tensors
        '''

        # Get the correct dataloader (test or train)
        data_loader = (
            self.train_dataloader
            if split == 'train'
            else self.test_dataloader
        )

        # Get the input and target tensors, and move to the right device
        for input, target in data_loader:
            input = input.to(self.config.device)
            target = target.to(self.config.device)

        # Return the input and target tensors (a batch of data)
        return input, target
