'''
The class for managing the dataset
'''

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformer_blocks import GPTConfig
from torch.nn.utils.rnn import pad_sequence
from tokenizer import ChessTokenizer

import os
import re
import json
from tqdm import tqdm

from typing import Tuple, Generator


class CreateDataSet():
    # Create a dataset from a given list of JSON files

    def __init__(
        self,
        file_list: list,
        batch_size: int,
        tokenizer: ChessTokenizer,
        min_length: int = 6,
        block_size: int = 192,
    ) -> None:
        # Setup basic values
        # Get a list of files to process

        self.file_list = file_list
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.min_length = min_length

        self.pad_id = 2

        self.train_dataset = None
        self.test_dataset = None
        self.skipped_games = 0

    def __enter__(self) -> None:
        # Context manager for the dataset

        # Collect games from JSON files
        game_list = self._collect_games()

        # Preprocess and tokenize the games
        processed_games = self._preprocess(game_list)

        # Split the games into training and testing sets
        train_data, test_data, train_labels, test_labels = self._split(
            processed_games
        )

        # Create dataloaders for the training and testing sets
        self._create_dataloaders(
            train_data,
            test_data,
            train_labels,
            test_labels
        )

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb
    ) -> None:
        # Context manager for the dataset
        # Return the two datasets

        return self.train_dataset, self.test_dataset

    def _collect_games(
        self,
    ) -> list:
        # Load JSON files, one at a time
        # Extract games from JSON files
        # Return a list of games as strings

        game_list = []
        for file in tqdm(self.file_list):
            # Load the JSON file
            with open(f"{self.dataset_dir}/{file}", "r") as f:
                game_file = json.load(f)
            
            # Get the game year (there is one per file)
            year = list(game_file.keys())[0]

            # Get each game (organized by month)
            for month in game_file[year]:
                for game in game_file[year][month]:
                    # Add the game to the list
                    game_list.append(game['pgn'])

        # A complete list of all games in the gives dataset
        return game_list

    def _preprocess(
        self,
        game_list: list,
    ) -> list:
        # Takes a list of games as strings
        # Cleanup game strings
        # Tokenize game strings
        # Returns a list of tokenized games (list of lists)

        processed_games = []
        for game in game_list:
            # Remove move numbers from the PGN
            game = re.sub(r"\d{1,3}\. ", "", game).strip()

            # Tokenize the game
            tokenized_game = self.tokenizer.tokenize(game, pad=False)

            # Skip games that are too short or too long
            if self.min_length <= len(tokenized_game) <= self.block_size:
                self.skipped_games += 1
                continue

            # Add the tokenized game to the list
            processed_games.append(tokenized_game)

        return processed_games

    def _split(
        self,
        processed_games: list,
        split: float = 0.2,
    ) -> Tuple[list, list, list, list]:
        # Takes a split value and a list of processed games
        # Split a list of games into training and testing lists
        #   (train_split, test_split)
        # Create input and target sequences from these two lists
        # Result: 4 lists (train_data, test_data, train_labels, test_labels)

        # Get input and target sequences
        #   Target is input shifted right by one token
        input_sequences = []
        target_sequences = []

        for game in processed_games:
            input_sequences.append(game[:-1])
            target_sequences.append(game[1:])

        # Split input_sequences and target_sequences into train and test sets
        (
            train_data, test_data, train_labels, test_labels
        ) = train_test_split(
            input_sequences,
            target_sequences,
            test_size=split,
        )

        return train_data, test_data, train_labels, test_labels

    def _create_dataloaders(
        self,
        train_data: list,
        test_data: list,
        train_labels: list,
        test_labels: list,
    ) -> None:
        # Takes the four lists
        # Combine four lists into two TensorDatasets
        # Create DataLoader objects for the training and testing sets,
        #   use collate_fn
        # Store the DataLoader objects in self

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
        #   Pin memory to speed up data transfer to the GPU
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )

    def _collate_fn(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Takes a batch of tensors
        # Add padding to the sequences
        # Returns a batch of padded sequences and labels

        batch = pad_sequence(
            batch,
            batch_first=True,
            padding_value=self.pad_id
        )

        return batch[:, :-1], batch[:, 1:]


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

        # Load the dataset
        self.load(
            min_moves=6,
            max_moves=190,
        )

        # Split the dataset
        self.split(
            test_size=0.2
        )

        # Create dataloaders
        self.create_dataloaders()

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
                            if len(game.split(" ")) < min_moves:
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
        #   Pin memory to speed up data transfer to the GPU
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            pin_memory=True,
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

    def _collate_fn(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Collate function for the DataLoader

        Practically, this doesn't do much, as the data is already padded
            This can be modified in future
        '''

        # Batch is a list of tuples (sequence, label)
        #   We need to convert this to a tuple of sequences and labels
        sequences, labels = zip(*batch)

        # Pad the sequences to have the same length
        #   Output is a tensor (shape is batch size, block size)
        sequences_padded = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=2
        )

        # Convert labels to a tensor
        labels_padded = pad_sequence(
            labels,
            batch_first=True,
            padding_value=2
        )

        return sequences_padded, labels_padded


if __name__ == '__main__':
    print("This is a module for managing the dataset")
    print("Please use 'model.py' for the main program")
