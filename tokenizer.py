'''
The tokenizer
This collects a list of known chess moves in standard chess notation

This needs to be word-based, not character-based or sub-word
    This is because we can not change the chess moves or invent new ones
'''

from typing import List
import json
import re
from tqdm import tqdm
import os
import random

import cProfile
import pstats
import time


CHECKPOINT = 50
PROFILE = False


class ChessTokenizer:
    '''
    Tokenizer for chess moves
        The 'words' are really chess moves in standard chess notation
        Rather than learning english, we are learning chess moves

    Uses custom word-based tokenization for chess moves
        We can't 'invent' different words, as the moves are standard

    Methods:
        __init__: Initializes the tokenizer
        __len__: Returns the size of the vocabulary
        train: Creates a mapping between words and indices
        learn_tokens: Learns the tokens from the input text
        json_save: Saves the mappings to JSON files
        load: Loads the mappings from JSON files
        save_resume: Saves the current state of the tokenizer to a resume file
        tokenize: Tokenizes the input text into a list of integers
        detokenize: Converts a list of integers (tokens) into a string
        pgn_extract: Parse the JSON files
    '''

    def __init__(
        self
    ) -> None:
        '''
        next_value is used to track the next item in the dict during training
            This is an optimization as training can take a very long time
        '''

        # A counter to track the next item in the dictionary
        self.next_value = 1

        # The padding token
        self.pad = "[Pad]"
        self.pad_number = 2

    def __len__(
        self
    ) -> int:
        '''
        Returns the size of the vocabulary, i.e., the number of unique tokens.
        This allows us to call len(tokenizer) to get
            the size of the vocabulary.

        Returns:
            The number of unique tokens in the vocabulary.
        '''

        return len(self.word2idx)

    def train(
        self,
        dataset_path: str = './dataset',
        save_path: str = '.',
        resume_file: str = f"{os.getcwd()}/resume.txt",
        resume: bool = False,
        percent: int = 100
    ) -> None:
        '''
        Main training function

        Creates a mapping between words and indices
        Creates a mapping between indices and words
        Save the mappings to JSON files
        Manages resuming training if needed
        Can train on a subset of the files if needed for testing

        Args:
            dataset_path: Path to the dataset
            save_path: Path to save the JSON files
            resume_file: Path to the resume file
            resume: Flag to resume training
            percent: Percentage of files to use for training
        '''

        # Get a list of all training files
        full_file_list = os.listdir(dataset_path)

        # Check if we can resume training
        if resume and os.path.exists(resume_file):
            self.load(save_path)

            # Read the resume file
            try:
                with open(os.path.join(resume_file), "r") as f:
                    resume_list = [line.strip() for line in f]
            except Exception as e:
                print(f"Error reading resume file: {e}")
                return

            # Remove files from full_file_list if they exist in 'resume'
            full_file_list = [
                file
                for file in full_file_list
                if file not in resume_list
            ]
            if len(full_file_list) == 0:
                print("No files to train on. Exiting.")
                return

            trained_files = []

        # Create dictionaries, and add special tokens
        #   This is only done once, so we check the flag
        else:
            self.word2idx = {
                "[Start]": 0,
                "[End]": 1,
                "[Pad]": 2,
                "[Unk]": 3,
                "[Mask]": 4,
            }
            trained_files = []
            try:
                with open(resume_file, "w") as f:
                    f.write("")
            except Exception as e:
                print(f"Error writing resume file: {e}")
                return

        # Track the next value for the dictionary
        self.next_value = max(self.word2idx.values()) + 1

        # Select the files to use for training
        random.shuffle(full_file_list)
        file_list = full_file_list[:int(len(full_file_list) * (percent / 100))]
        print(
            f"Using {len(file_list)} out of {len(full_file_list)} files\
            for tokenizer training."
        )

        # Create the file list
        file_list = [
            os.path.join(dataset_path, file)
            for file in file_list
            if os.path.isfile(os.path.join(dataset_path, file))
        ]

        # Train the tokenizer
        for index, file in enumerate(
            tqdm(
                file_list,
                desc="Total Progress",
                colour="green"
            )
        ):
            moves = self._pgn_extract(file)
            for idx, _ in enumerate(
                tqdm(
                    moves,
                    desc="File Progress",
                    leave=False,
                    colour="yellow",
                )
            ):
                # Learn the tokens
                self._learn_tokens(moves[idx].split(" "))

            # Keep track of the files trained, enabling resuming
            trained_files.append(file.split("\\")[-1])
            if (
                (len(trained_files) == CHECKPOINT) or
                (index == len(file_list) - 1)
            ):
                self._save_resume(
                    resume_file,
                    trained_files
                )
                self._save(
                    self.word2idx,
                    self.idx2word,
                    save_path,
                )
                trained_files = []

        # Save the mappings to JSON files
        self._save(
            self.word2idx,
            self.idx2word,
            save_path
        )

    def _learn_tokens(
        self,
        moves: List[str]
    ) -> None:
        '''
        Learn the tokens from the input text
        This is a separate method to allow parallelism
            However, experimentally, this is not faster

        Args:
            moves: List of chess moves as strings
        '''

        # Create temporary set and dictionary
        temp_word2idx = {}
        temp_tokens = set()

        # Learn tokens
        for token in moves:
            if token not in temp_tokens:
                idx = len(temp_word2idx)
                temp_word2idx[token] = idx
                temp_tokens.add(token)

        # Merge forward mappings (unique values only)
        for key in temp_word2idx.keys():
            if key not in self.word2idx:
                self.word2idx[key] = self.next_value
                self.next_value += 1

        # Create the reverse mapping (int: str)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def _save(
        self,
        word2idx: dict,
        idx2word: dict,
        save_path: str = '.',
    ) -> None:
        '''
        Save the mappings to JSON files
        If the files already exist, they will be overwritten

        Args:
            word2idx: Dictionary mapping words to indices
            idx2word: Dictionary mapping indices to words
            save_path: Path to save the JSON files
        '''

        # Create the save path
        word2idx_path = f"{save_path}\\word2idx.json"
        idx2word_path = f"{save_path}\\idx2word.json"

        # Save the files
        try:
            with open(word2idx_path, "w") as f:
                json.dump(word2idx, f)
        except Exception as e:
            print(f"Error saving word2idx: {e}")
            return

        try:
            with open(idx2word_path, "w") as f:
                json.dump(idx2word, f)
        except Exception as e:
            print(f"Error saving idx2word: {e}")
            return

    def load(
        self,
        path: str = '.'
    ) -> None:
        '''
        Load the mappings from JSON files
        JSON files use strings, so the keys need to be converted to integers

        Args:
            path: Path to the JSON files
        '''

        # Open the files, and load the forward and reverse mappings
        try:
            with open(os.path.join(path, "word2idx.json"), "r") as f:
                self.word2idx = json.load(f)
        except Exception as e:
            print(f"Error loading word2idx: {e}")
            return

        try:
            with open(os.path.join(path, "idx2word.json"), "r") as f:
                self.idx2word = json.load(f)
        except Exception as e:
            print(f"Error loading idx2word: {e}")
            return

        # Convert string keys to integer
        self.idx2word = {int(k): v for k, v in self.idx2word.items()}

    def _save_resume(
        self,
        path: str,
        files: List[str]
    ) -> None:
        '''
        Save the current state of the tokenizer to a resume file

        Args:
            path: Path to the resume file
        '''

        # Just open the resume file and append the files
        try:
            with open(path, "a") as f:
                for file in files:
                    f.write(f"{file}\n")
        except Exception as e:
            print(f"Error saving resume file: {e}")
            return

    def tokenize(
        self,
        text: str,
        pad: bool = False,
        pad_size: int = 0
    ) -> List[int]:
        '''
        Tokenizes the input text into a list of integers

        (1) Splits the text into words (chess moves)
        (2) Converts the words into indices using the word2idx mapping

        Args:
            text: Input text to tokenize
                Space separated chess moves

        Returns:
            List of integers, where each integer represents a chess move
        '''

        tokens = text.split()
        token_ids = [
            # Prepend the [Start] token's ID
            self.word2idx['[Start]'],
            # (3) Puts the token into the list
            *[
                self.word2idx[token]
                # (1) Iterate over the tokens
                for token in tokens
                # (2) Check if the token is in the dict
                if token in self.word2idx
            ],
            # Append the [End] token's ID
            self.word2idx['[End]']
        ]

        if pad:
            token_ids += (
                [self.word2idx[self.pad]] * (pad_size - len(token_ids))
            )

        return token_ids

    def detokenize(
        self,
        token_ids: List[int]
    ) -> str:
        '''
        Converts a list of integers (tokens) into a string

        (1) Converts the indices into words using the idx2word mapping
        (2) Joins the words into a single string

        Args:
            token_ids: List of integers, representing a chess move

        Returns:
            Single string representing the chess moves
        '''

        # Filter out the token IDs for [Start] and [End] tokens
        filtered_token_ids = [
            token_id for token_id in token_ids
            if token_id not in (
                self.word2idx['[Start]'],
                self.word2idx['[End]']
            )
        ]

        tokens = [
            # (3) Adds the token to the list
            self.idx2word[token_id]
            # (1) Looks through the tokens list
            for token_id in filtered_token_ids
            # (2) Checks if the token is in the dict
            if token_id in self.idx2word
        ]

        # Convert back to a string, separated by spaces
        text = " ".join(tokens)

        return text

    def _pgn_extract(
        self,
        file_path: str
    ) -> List[str]:
        '''
        Parse the JSON files
        This creates a list of moves without the metadata or move number
            Basically, a space-delimited list of tokens

        Args:
            file_path: Path to the JSON file

        Returns:
            List of chess moves as strings
                Each string is a complete game
        '''

        try:
            with open(file_path, "r") as f:
                moves = json.load(f)
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return []

        # A list of complete games
        game_list = []

        # Parse the JSON file for games
        year = list(moves.keys())[0]
        for month in moves[year]:
            for game in moves[year][month]:
                game = re.sub(r"\d{1,3}\. ", "", game['pgn'])
                game_list.append(game.strip())

        return game_list


def confirm_mappings(
    tokenizer: ChessTokenizer
) -> bool:
    '''
    A simple function for debugging
    This confirms that the forward and reverse mappings align correctly
    '''

    for word, idx in tokenizer.word2idx.items():
        # Use the idx to get the corresponding word in idx2word
        reverse_lookup_word = tokenizer.idx2word.get(idx)

        # Check if the word matches the reverse lookup
        if word != reverse_lookup_word:
            print(f"Mismatch found: {word} -> {idx} -> {reverse_lookup_word}")
            return False

    print("All mappings and reverse mappings correspond correctly.")
    return True


def main():
    '''
    Main function to train the tokenizer when run as a script
    This exists for debugging purposes, and to allow profiling
    Set PROFILE to True to enable profiling
    '''

    start = time.perf_counter()

    # Create the tokenizer
    tokenizer = ChessTokenizer()

    # Train the tokenizer
    tokenizer.train(
        resume=True,
        percent=20
    )

    # Check the mappings
    confirm_mappings(tokenizer)

    finish = time.perf_counter()
    print(f"Finished in {finish - start:.2f} seconds.")


if __name__ == "__main__":
    '''
    Only used if this is run as a script
    Set PROFILE to True to enable profiling
    '''

    if PROFILE:
        cProfile.run('main()', 'profiling_results.stats')
        p = pstats.Stats('profiling_results.stats')
        p.sort_stats('cumulative').print_stats(10)

    else:
        main()
