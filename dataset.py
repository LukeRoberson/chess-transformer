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

from typing import Tuple, Generator


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

        data_iter:
            Iterate over the dataset in batches
    '''

    def __init__(
        self,
        model_config: GPTConfig,
        train_config,
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
        self.model_config = model_config
        self.config = train_config
        self.dataset_dir = dataset_dir
        self.tokenizer = model_config.tokenizer

        # Important values to be set later
        self.padded_game_list = None
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.train_dataloader = None
        self.test_dataloader = None

        # Iterator for the DataLoader
        self.data_loader_iter = None

    def load(
        self,
        min_moves: int = 6,
        max_moves: int = 190,
    ) -> None:
        '''
        Load the dataset from JSON files

        NOTE: Here we measure moves as each move a player makes
            This means each player's move is counted as a move
            Two moves makes one full turn
            Each move is one token in the dataset

        We need to track the longest game so we can later pad shorter games,
            so they're all the same length.
            This is needed to convert to a tensor later on.

        Args:
            min_moves: int
                The minimum number of moves a game must have to be included
            max_moves: int
                The maximum number of moves a game can have to be included
                Lowering this value can reduce the dataset size
                This also affects the block size of the model
                    Less moves = smaller block size = Less parameters
                    Remember to factor in start and end tokens

        Returns:
            int
                The length of the longest game in the dataset
                This can be used to set the block size of the model
        '''

        # A starting point for the game list
        game_list = []

        # Track the longest game
        max_length = 0

        # Track excluded games
        too_short = 0
        too_long = 0

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

                            # Skip extrememly short games
                            if len(game.split(" ")) < 6:
                                too_short += 1
                                continue

                            # Skip games that are quite long
                            if len(game.split(" ")) > max_moves:
                                too_long += 1
                                continue

                            # Add the game to the list
                            game_list.append(game)

                            # Find the maximum length of a game
                            if len(
                                self.tokenizer.tokenize(game)
                            ) > max_length:
                                max_length = len(
                                    self.tokenizer.tokenize(game)
                                )

                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error processing file {file}: {e}")

        print(f"Longest included game is {max_length} individual moves")
        print(
            f"Excluded {too_short} games that were too short "
            f"({(too_short/(len(game_list) + too_short + too_long))*100:.2f}%)"
        )
        print(
            f"Excluded {too_long} games that were too long "
            f"({(too_long/(len(game_list) + too_short + too_long))*100:.2f}%)"
        )

        # Tokenize and pad each game list to the maximum length
        #   This means all lists are the same size
        #   This is needed to convert to tensor later
        self.padded_game_list = [
            self.tokenizer.tokenize(game, pad=True, pad_size=max_length)
            for game in game_list
        ]

        game_size = len(self.padded_game_list)
        print(f"Loaded {game_size} games into the dataset.")
        print(
            f"Training on {int(game_size * (1.0 - self.config.test_split))}"
            " games"
        )

        return max_length

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
            batch_size=self.model_config.batch_size,
            shuffle=shuffle
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False
        )

    def get_batch(
        self,
        split: str = 'train'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Get a single batch of data from the training or testing set
        This returns a batch of input and target tensors
            They are automatically moved to the right device

        Args:
            split: str
                The split to get the batch from
                Either 'train' or 'test'

        Returns:
            Tuple[torch.Tensor, torch.Tensor] or None
                The input and target tensors
                None if all batches are exhausted
        '''

        # Get the correct dataloader (test or train) as an iterator
        data_loader = (
            self.train_dataloader
            if split == 'train'
            else self.test_dataloader
        )
        data_loader_iter = iter(data_loader)

        # Get the next batch, or return None if the iterator is exhausted
        batch = next(data_loader_iter, None)

        # If the iterator is exhausted and returned None, return None
        if batch is None:
            return None

        # Get the input and target tensors, and move to the right device
        input, target = batch
        input = input.to(self.model_config.device)
        target = target.to(self.model_config.device)

        # Return the input and target tensors (a batch of data)
        return input, target

    def data_iter(self, split: str = 'train') -> Generator:
        '''
        Iterate over the dataset

        Creates a generator that yields batches of data and targets at a time.
            This makes the function iterable, so it can be used in a for loop.

        This partitions the data into batches, allowing for processing large
            datasets in manageable chunks without loading the entire dataset
            into memory at once, enhancing memory efficiency.

        Args:
            split: str
                The split to iterate over
                Either 'train' or 'test'

        Yields:
            Tuple[List[Any], List[Any]]
        '''

        # Choose the correct dataloader
        if split == 'train':
            data_loader = self.train_dataloader
        else:
            data_loader = self.test_dataloader

        # Initialize the iterator if not already done or if it's exhausted
        if self.data_loader_iter is None:
            self.data_loader_iter = iter(data_loader)

        while True:
            try:
                batch = next(self.data_loader_iter)
                input, target = batch
                yield input, target

            # This handles running out of data in the iterator
            except StopIteration:
                self.data_loader_iter = iter(data_loader)
                break


if __name__ == '__main__':
    print("This is a module for managing the dataset")
    print("Please use 'model.py' for the main program")
