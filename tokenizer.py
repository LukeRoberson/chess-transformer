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
import cProfile
import pstats
import time


class ChessTokenizer:
    '''
    Tokenizer for chess moves
        The 'words' are really chess moves in standard chess notation
        Rather than learning english, we are learning chess moves

    Uses custom word-based tokenization for chess moves
        We can't 'invent' different words, as the moves are standard

    Methods:
        train: Creates a mapping between words and indices
        learn_tokens: Learns the tokens from the input text
        json_save: Saves the mappings to JSON files
        load: Loads the mappings from JSON files
        tokenize: Tokenizes the input text into a list of integers
        detokenize: Converts a list of integers (tokens) into a string
    '''

    def __init__(self):
        '''
        The constructor
        The train_prep flag is used to check if the mappings have been started
            This allows training to resume
        '''

        # Set the train_prep flag
        self.train_prep = False

        # A counter to track the next item in the dictionary
        self.next_value = 1

    def train(self, file_list: List[str]):
        '''
        Main training function

        Creates a mapping between words and indices
        Creates a mapping between indices and words
        Save the mappings to JSON files

        Args:
            moves: List of chess moves as strings
                In standard chess notation, space separated
        '''

        # Create dictionaries, and add special tokens
        #   This is only done once, so we check the flag
        if self.train_prep is False:
            self.word2idx = {
                "[Open]": 0,
                "[CheckMate]": 1,
                "[Draw]": 2,
                "[Resign]": 3,
                "[StaleMate]": 4,
            }
            self.train_prep = True

        # Track the next value for the dictionary
        self.next_value = max(self.word2idx.values()) + 1

        # Train the tokenizer
        for file in tqdm(file_list, desc="Total Progress", colour="green",):
            moves = pgn_extract(file)
            for idx, _ in enumerate(
                tqdm(
                    moves,
                    desc="File Progress",
                    leave=False,
                    colour="yellow",
                )
            ):
                # Learn the tokens
                self.learn_tokens(moves[idx].split(" "))

        # Save the mappings to JSON files
        self.json_save(self.word2idx, self.idx2word)

    def learn_tokens(self, moves: List[str]):
        '''
        Learn the tokens from the input text
        This is a separate method to allow parallelism

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

    def json_save(self, word2idx: dict, idx2word: dict):
        '''
        Save the mappings to JSON files

        Args:
            word2idx: Dictionary mapping words to indices
            idx2word: Dictionary mapping indices to words
        '''

        with open("word2idx.json", "w") as f:
            json.dump(word2idx, f)

        with open("idx2word.json", "w") as f:
            json.dump(idx2word, f)

    def load(self):
        '''
        Load the mappings from JSON files
        JSON files use strings, so the keys need to be converted to integers
        '''

        with open("word2idx.json", "r") as f:
            self.word2idx = json.load(f)

        with open("idx2word.json", "r") as f:
            self.idx2word = json.load(f)

        # Convert string keys to integer
        self.idx2word = {int(k): v for k, v in self.idx2word.items()}

        # Set the train_prep flag
        self.train_prep = True

    def tokenize(self, text: str) -> List[int]:
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
            # (3) Puts the token into the list
            self.word2idx[token]
            # (1) Iterate over the tokens
            for token in tokens
            # (2) Check if the token is in the dict
            if token in self.word2idx
        ]

        return token_ids

    def detokenize(self, token_ids: List[int]) -> str:
        '''
        Converts a list of integers (tokens) into a string

        (1) Converts the indices into words using the idx2word mapping
        (2) Joins the words into a single string

        Args:
            token_ids: List of integers, representing a chess move

        Returns:
            Single string representing the chess moves
        '''

        tokens = [
            # (3) Adds the token to the list
            self.idx2word[token_id]
            # (1) Looks through the tokens list
            for token_id in token_ids
            # (2) Checks if the token is in the dict
            if token_id in self.idx2word
        ]

        # Convert back to a string, separated by spaces
        text = " ".join(tokens)

        return text


def pgn_extract(file_path: str):
    '''
    Parse the JSON files
    This creates a list of moves without the metadata or move number

    Args:
        file_path: Path to the JSON file

    Returns:
        List of chess moves as strings
            Each string is a complete game
    '''

    with open(file_path, "r") as f:
        moves = json.load(f)

    # A list of complete games
    game_list = []

    # Parse the JSON file for games
    year = list(moves.keys())[0]
    for month in moves[year]:
        for game in moves[year][month]:
            game = re.sub(r"\d{1,3}\. ", "", game['pgn'])
            game_list.append(game.strip())

    return game_list


def confirm_mappings(tokenizer):
    for word, idx in tokenizer.word2idx.items():
        # Use the idx to get the corresponding word in idx2word
        reverse_lookup_word = tokenizer.idx2word.get(idx)

        # Check if the word matches the reverse lookup
        if word != reverse_lookup_word:
            print(f"Mismatch found: {word} -> {idx} -> {reverse_lookup_word}")
            return False

    print("All mappings and reverse mappings correspond correctly.")
    return True


# Wrap your main code block in a function for profiling
def main():
    start = time.perf_counter()

    # Create the tokenizer
    tokenizer = ChessTokenizer()

    # Get a list of files to train with
    dataset_dir = "./dataset"
    file_list = os.listdir(dataset_dir)
    file_list = [
        os.path.join(dataset_dir, file)
        for file in file_list
        if os.path.isfile(os.path.join(dataset_dir, file))
    ]

    # Train the tokenizer
    tokenizer.train(file_list)

    # Check the mappings
    confirm_mappings(tokenizer)

    finish = time.perf_counter()
    print(f"Finished in {finish - start:.2f} seconds.")
    # 4 files finished in 14.55 seconds, no parallelism
    # 4 files finish in 0.88 seconds, no parallelism, JSON save moved outside of loop
    #   10 in 4.82 seconds
    #   50 in 32.63 seconds
    # 50 in 31.16 seconds with optimized max() calls


if __name__ == "__main__":
    # Run normally
    # main()

    # Use profiling to analyze stats
    cProfile.run('main()', 'profiling_results.stats')
    p = pstats.Stats('profiling_results.stats')
    p.sort_stats('cumulative').print_stats(10)
