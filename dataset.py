'''
The class for managing the dataset
'''

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformer_blocks import GPTConfig
from tokenizer import ChessTokenizer

import os
import re
import json
from tqdm import tqdm
import traceback

from typing import Tuple, Generator


class CreateDataSet():
    '''
    Creates two dataloaders, given a list of JSON files
    Supports being used as a context manager

    Methods:
        __init__:
            Constructor

        __enter__:
            Context manager for the dataset

        __exit__:
            Context manager for the dataset

        _collect_games:
            Collect games from JSON files

        _preprocess:
            Preprocess and tokenize the games

        _split:
            Split the games into training and testing sets

        _create_dataloaders:
            Create dataloaders for the training and testing sets
    '''

    def __init__(
        self,
        file_list: list,
        batch_size: int,
        tokenizer: ChessTokenizer,
        min_length: int = 6,
        block_size: int = 192,
    ) -> None:
        '''
        Constructor
        Sets up the values for the dataset
        Initializes needed objects

        Args:
            file_list: list
                A list of JSON files to load games from
            batch_size: int
                The batch size for the dataloaders
            tokenizer: ChessTokenizer
                The tokenizer for the dataset
            min_length: int
                The minimum length of a game to be included
            block_size: int
                The block size for the model
                Also the maximum length of a game in tokens
        '''

        # Setup the values for the dataset
        self.file_list = file_list
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.min_length = min_length
        self.pad_id = tokenizer.pad_number

        # Initialize the dataloaders
        self.train_dataloader = None
        self.test_dataloader = None
        self.train_data_size = None
        self.test_data_size = None

        # Track the number of skipped games
        self.skipped_games = 0

    def __enter__(
        self
    ) -> Tuple[DataLoader, DataLoader, int, int]:
        '''
        Context manager for the dataset
        This is the main flow of the dataset creation

        (1) Collect games from JSON files
        (2) Preprocess and tokenize the games
        (3) Split the games into training and testing sets
        (4) Create dataloaders for the training and testing sets

        Returns:
            Tuple[DataLoader, DataLoader, int, int]
                The training and testing dataloaders
                The sizes of the training and testing sets
        '''

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

        # Return the dataloaders and the sizes of the training and testing sets
        return (
            self.train_dataloader,
            self.test_dataloader,
            self.train_data_size,
            self.test_data_size
        )

    def __exit__(
        self,
        exc_type: type,
        exc_val: Exception,
        exc_tb: traceback,
    ) -> bool:
        '''
        Context manager for the dataset
        Handles exceptions and cleans up resources

        Args:
            exc_type: type
                The type of the exception
            exc_val: Exception
                The exception object
            exc_tb: traceback
                The traceback object

        Returns:
            bool
                Whether to propagate the exception or not
        '''

        # Handle exceptions
        if exc_type is not None:
            print(f"An error occurred: {exc_val}")
            # Print the traceback if there's an exception
            traceback.print_tb(exc_tb)

        # Clean up resources
        self.train_dataloader = None
        self.test_dataloader = None
        self.train_data_size = None
        self.test_data_size = None

        # Propagate exception if there is one
        return False

    def _collect_games(
        self,
    ) -> list:
        '''
        Collect games from JSON files

        (1) Load JSON files, one at a time
        (2) Extract games from JSON files
        (3) Return a list of games as strings

        Returns:
            list
                A list of raw games as strings
        '''

        # Loop through each JSON file and extract the games
        game_list = []
        for file in tqdm(self.file_list):
            # Load the JSON file
            with open(file, "r") as f:
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
        '''
        Preprocess and tokenize the games

        (1) Cleanup game strings (eg, remove move numbers)
        (2) Tokenize and pad the game strings
        (3) Return a list of tokenized games (list of lists)

        Args:
            game_list: list
                A list of raw games as strings

        Returns:
            list
                A list of tokenized games as lists
        '''

        # Loop through each game and preprocess it
        processed_games = []
        for game in game_list:
            # Remove move numbers from the PGN
            game = re.sub(r"\d{1,3}\. ", "", game).strip()

            # Tokenize the game, and pad the sequence
            tokenized_game = self.tokenizer.tokenize(
                game,
                pad=True,
                pad_size=self.block_size
            )

            # Skip games that are too short or too long
            if (
                len(tokenized_game) < self.min_length
                or len(tokenized_game) > self.block_size
            ):
                self.skipped_games += 1
                continue

            # Add the tokenized game to the list
            processed_games.append(tokenized_game)

        # Print some interesting stats
        print(f"skipped {self.skipped_games} games due to size")
        print(f"{len(processed_games)} games pre-processed")

        return processed_games

    def _split(
        self,
        processed_games: list,
        split: float = 0.2,
    ) -> Tuple[list, list, list, list]:
        '''
        Split the games into training and testing sets

        (1) Create input and target sequences from the games
            The target is the input shifted right by one token
        (2) Split the sequences into training and testing sets
        (3) Return the training and testing sets

        Args:
            processed_games: list
                A list of tokenized games as lists
            split: float
                The percentage size of the test set
                The remainder is the training set

        Returns:
            Tuple[list, list, list, list]
                The training and testing sets of input and target sequences
        '''

        input_sequences = []
        target_sequences = []

        # Get input and target sequences
        #   Target is input shifted right by one token
        for game in processed_games:
            input_sequences.append(game[:-1])
            target_sequences.append(game[1:])

        # Split input_sequences and target_sequences into train and test sets
        train_data, test_data, train_labels, test_labels = train_test_split(
            input_sequences,
            target_sequences,
            test_size=split,
        )

        # Store the sizes of the training and testing sets
        #   The scheduler needs this information later
        self.train_data_size = len(train_data)
        self.test_data_size = len(test_data)

        # Return four lists
        return train_data, test_data, train_labels, test_labels

    def _create_dataloaders(
        self,
        train_data: list,
        test_data: list,
        train_labels: list,
        test_labels: list,
    ) -> None:
        '''
        Create dataloaders for the training and testing sets

        (1) Combine four lists into two TensorDatasets
        (2) Create DataLoader objects for the training and testing sets

        Args:
            train_data: list
                The training input sequences
            test_data: list
                The testing input sequences
            train_labels: list
                The training target sequences (labels)
            test_labels: list
                The testing target sequences (labels)
        '''

        # Create TensorDatasets for the training and testing sets
        train_dataset = TensorDataset(
            torch.tensor(train_data),
            torch.tensor(train_labels)
        )
        test_dataset = TensorDataset(
            torch.tensor(test_data),
            torch.tensor(test_labels)
        )

        # Create DataLoader objects for the training and testing sets
        #   Pin memory to speed up data transfer to the GPU
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )


class ManageDataSet():
    '''
    The main class for managing the dataloaders
    Uses the CreateDataSet class to create the dataloaders

    Methods:
        __init__:
            Request dataloaders from the CreateDataSet class

        get_batch:
            Get a batch of data from the training or testing set

        data_iter:
            Iterate over the dataset in batches
    '''

    def __init__(
        self,
        model_config: GPTConfig,
        # train_config,
        dataset_dir: str = './dataset'
    ) -> None:
        '''
        Constructor

        (1) Setup initial values
        (2) Collect JSON files from the dataset directory
        (3) Create a dataset using the context manager

        Args:
            config: GPTConfig
                The configuration for the GPT model

            train_config: dict
                The configuration for training the model

            dataset_dir: str
                The directory containing the dataset
        '''

        # Required initial values
        self.model_config = model_config
        self.dataset_dir = dataset_dir
        self.tokenizer = model_config.tokenizer

        # Important values to be set later
        self.train_dataloader = None
        self.test_dataloader = None
        self.train_data_size = None
        self.test_data_size = None

        # Iterator for the DataLoader
        self.data_loader_iter = None

        # Get a list of all files in the dataset directory
        file_list = []
        for file in tqdm(
            os.listdir(self.dataset_dir),
            desc='Collecting JSON files'
        ):
            if file.endswith('.json'):
                # Add to file list
                file_list.append(f"{self.dataset_dir}/{file}")

        # Create a dataset using the context manager
        with CreateDataSet(
            file_list=file_list,
            batch_size=self.model_config.batch_size,
            tokenizer=self.tokenizer,
            min_length=6,
            block_size=192,
        ) as (train_dataset, test_dataset, train_size, test_size):
            self.train_dataloader = train_dataset
            self.test_dataloader = test_dataset
            self.train_data_size = train_size
            self.test_data_size = test_size

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
