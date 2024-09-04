'''
The classes for creating and managing the datasets

CreateDataSet:
    Creates dataloaders for training and testing sets

ManageDataSet:
    Manages the dataloaders for the model
    Includes getting batches, and chunking the raw data
'''

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformer_blocks import GPTConfig
from tokenizer import ChessTokenizer

import os
import re
import json
from tqdm import tqdm
import random
import numpy as np
import gc
import threading
import queue
import time

from typing import Tuple, Generator
import traceback


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

        _loader:
            Load a single JSON file and return the contents

        _load_files:
            Generator to load files one at a time

        _load_games:
            Generator to load games one at a time

        _build_dataset:
            Build the dataset
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
        self.train_data_size = None

        # Track the number of skipped games
        self.skipped_games = 0

    def __enter__(
        self
    ) -> Tuple[DataLoader, int]:
        '''
        Context manager for the dataset
        This is the main flow of the dataset creation

        Returns:
            Tuple[DataLoader, DataLoader, int, int]
                The training dataloader
                The size of the training set
        '''

        # Create the dataset and dataloaders
        self._build_dataset()

        # Return the dataloaders and the sizes of the training and testing sets
        return (
            self.train_dataloader,
            self.train_data_size,
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

    def _loader(
        self,
        file: str,
    ) -> dict:
        '''
        Load a single JSON file and return the contents

        Args:
            file: str
                The path to the JSON file

        Returns:
            dict
                The contents of the JSON file
        '''

        with open(file, 'r') as f:
            contents = json.load(f)
            year = list(contents.keys())[0]
            return contents[year]

    def _load_files(
        self,
        files: list,
    ) -> Generator:
        '''
        Generator to load files one at a time

        Args:
            files: list
                A list of JSON files to load

        Yields:
            dict
                The contents of the JSON file
        '''

        for file in files:
            yield self._loader(file)

    def _load_games(
        self,
        games: dict,
    ) -> Generator:
        '''
        Generator to load games one at a time
        Each file has one or more games,
            this function yields them one at a time

        Args:
            games: dict

        Yields:
            np.array
                A tokenized game as a numpy array
        '''

        # Loop through each month in the games
        for month in games:
            # Loop through each game in the month
            for game in games[month]:
                # Strip the move numbers from the PGN
                filtered = re.sub(r"\d{1,3}\. ", "", game['pgn']).strip()
                length = len(filtered.split(" "))

                # Skip games that are too short or too long
                if length > self.min_length and length < (self.block_size - 1):
                    # Tokenize the game and pad the sequence
                    processed = self.tokenizer.tokenize(
                        filtered,
                        pad=True,
                        pad_size=self.block_size,
                    )

                    # Yield the tokenized game
                    yield np.array(processed)

    def _build_dataset(
        self,
    ) -> None:
        '''
        Build the dataset

        (1) Loads files and games as needed
        (2) Formats into a 2D Numpy array
        (3) Converts to input and target sequences
        (4) Splits into training and testing sets
        (5) Creates DataLoader objects for the training and testing sets

        Uses numpy arrays for memory efficiency
        Adds manual garbage collection to help with memory management
        '''

        np_array = np.empty((0, self.block_size), dtype=object)

        # Get games by year
        for year in tqdm(
            self._load_files(self.file_list),
            desc='Pre-loading dataset',
            total=len(self.file_list),
            colour='cyan',
            leave=False,
        ):
            all_games = []

            # Build a list of games (np arrays)
            for game in self._load_games(year):
                all_games.append(game)

            # After each file, convert list to np array
            if np_array.size == 0:
                np_array = np.array(all_games)
            elif len(all_games) > 0:
                np_array = np.concatenate(
                    (np_array, np.array(all_games)),
                    axis=0
                )

        # Create input and target sequences
        train_data = np_array[:, :-1]
        train_labels = np_array[:, 1:]

        # Get the sizes of the training and testing sets
        self.train_data_size = len(train_data)

        # Create TensorDatasets for the training and testing sets
        train_dataset = TensorDataset(
            torch.tensor(train_data),
            torch.tensor(train_labels)
        )
        del train_data
        del train_labels
        gc.collect()

        # Create DataLoader objects for the training and testing sets
        #   Pin memory to speed up data transfer to the GPU
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        gc.collect()


class ManageDataSet():
    '''
    The main class for managing the dataloaders
    Uses the CreateDataSet class to create the dataloaders

    Threading enables parts of the dataset to be loaded in the background
        This means training can continue while the next dataset is prepared
        The dataset is loaded in chunks. Each chunk is a percentage
            of the entire dataset

    Methods:
        __init__:
            Request dataloaders from the CreateDataSet class

        get_dataset:
            Get a dataset with a certain percentage of the files

        data_loader_thread:
            Thread for loading the dataset in the background

        start_data_loading:
            Start the thread for loading the dataset

        stop_data_loading:
            Stop the thread for loading the dataset

        get_batch:
            Get a batch of data from the training or testing set

        data_iter:
            Iterate over the dataset in batches
    '''

    def __init__(
        self,
        model_config: GPTConfig,
        dataset_dir: str = './dataset'
    ) -> None:
        '''
        Constructor

        (1) Setup initial values
        (2) Collect JSON files from the dataset directory
        (3) Setup threading for loading the dataset
        (4) Create a dataset using the context manager

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
        original_file_list = []
        for file in os.listdir(self.dataset_dir):
            if file.endswith('.json'):
                # Add to file list
                original_file_list.append(f"{self.dataset_dir}/{file}")

        # File list is shuffled to help prevent overfitting
        random.shuffle(original_file_list)

        # Get 10% of the files
        num_files = int(len(original_file_list) * 0.1)
        self.eval_list = original_file_list[:num_files]
        self.train_list = original_file_list[num_files:]

        # Create a copy of the original file list that we can edit
        self.files_remaining = self.train_list.copy()

        # Setup threading
        self.data_queue = queue.Queue(maxsize=1)
        self.stop_loading = threading.Event()

    def get_dataset(
        self,
        percentage: float = 1.0,
    ) -> None:
        '''
        Get a dataset with a certain percentage of the files
        The percentage determines the chunk size
        While the model is training on one chunk, the next chunk is prepared

        This uses 'files_remaining' to keep track of which files are left
            from the entire dataset. This ensures that all files are used
            in the epoch
        When no files are left, the entire dataset has been processed
            This can be repeated for multiple epochs

        Uses the CreateDataSet class to create the dataloaders

        Args:
            percentage: float
                The percentage of the dataset to use in a chunk
        '''

        # Handle the case where the percentage is invalid
        if percentage > 1.0:
            percentage = 1.0
        elif percentage <= 0.0:
            percentage = 1.0

        # If there are no files left, evaluate model and return None
        if len(self.files_remaining) == 0:
            # Reset the files remaining to the original list
            self.files_remaining = self.train_list.copy()
            random.shuffle(self.files_remaining)
            return None

        # Get the number of files to use
        file_count = len(self.train_list) * percentage

        # Get the files to use, and remove them from the remaining list
        file_list = []
        for _ in range(int(file_count)):
            # Handle the case where there are no files left
            if not self.files_remaining:
                break

            # Remove and return the first element
            file_list.append(self.files_remaining.pop(0))

        # Create a dataset using the context manager
        with CreateDataSet(
            file_list=file_list,
            batch_size=self.model_config.batch_size,
            tokenizer=self.tokenizer,
            min_length=6,
            block_size=192,
        ) as (train_dataset, train_size):
            self.train_dataloader = train_dataset
            self.train_data_size = train_size

    def get_test_dataset(
        self,
    ):
        '''
        Get the test dataset for the model

        This is separate to the training dataset,
            as the training set is chunked.
        This means we can evaluate the model on a test set
            after the chunks are processed
        '''

        # Create the dataset as normal
        with CreateDataSet(
            file_list=self.eval_list,
            batch_size=self.model_config.batch_size,
            tokenizer=self.tokenizer,
            min_length=6,
            block_size=192,
        ) as (test_dataset, test_size):
            self.test_dataloader = test_dataset
            self.test_data_size = test_size

    def data_loader_thread(
        self,
        percentage: float
    ) -> None:
        '''
        Loads the dataset in the background using an existing thread
        Checks if the 'stop loading' event is set. If not, it loads the
            next dataset chunk.
        The next chunk is only loaded if the queue is empty
            This saves system memory usage by only loading chunks as needed

        Args:
            percentage: float
                The percentage of the dataset to use in a chunk
        '''

        # loop to load the dataset in the background
        while not self.stop_loading.is_set():
            # Check if the queue is empty
            if self.data_queue.empty():
                # Build a new dataset
                self.get_dataset(percentage)

                # Put the new dataset in the queue
                self.data_queue.put(
                    (
                        self.train_dataloader,
                        self.train_data_size,
                    )
                )

            # A short sleep before checking again
            time.sleep(1)

    def start_data_loading(
        self,
        percentage: float
    ):
        '''
        Creates a thread to load the dataset in the background

        Args:
            percentage: float
        '''

        # Create a thread to load the dataset in the background
        self.loading_thread = threading.Thread(
            target=self.data_loader_thread,
            args=(percentage,)
        )

        # Start the thread
        self.loading_thread.start()

    def stop_data_loading(
        self
    ) -> None:
        '''
        Stops the thread for loading the dataset in the background
        '''

        # Set the stop loading event to stop the thread
        self.stop_loading.set()

        # Wait for the thread to finish before continuing
        self.loading_thread.join()

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
