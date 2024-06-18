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


class ChessTokenizer:
    '''
    Tokenizer for chess moves
        The 'words' are really chess moves in standard chess notation
        Rather than learning english, we are learning chess moves

    Uses custom word-based tokenization for chess moves
        We can't 'invent' different words, as the moves are standard

    Methods:
        train: Creates a mapping between words and indices
        tokenize: Tokenizes the input text into a list of integers
        detokenize: Converts a list of integers (tokens) into a string
    '''

    def __init__(self):
        '''
        The constructor
        Nothing to do here
        '''
        pass

    def train(self, moves: List[str]):
        '''
        Creates a mapping between words and indices
        Creates a mapping between indices and words
        Save the mappings to JSON files

        Args:
            moves: List of chess moves as strings
                In standard chess notation, space separated
        '''

        self.moves = moves.split(" ")

        # Create dictionaries, and add special tokens
        #   This is only done once
        if hasattr(self, "word2idx") is False:
            self.word2idx = {
                "[Open]": 0,
                "[CheckMate]": 1,
                "[Draw]": 2,
                "[Resign]": 3,
                "[StaleMate]": 4,
            }

            self.idx2word = {
                0: "[Open]",
                1: "[CheckMate]",
                2: "[Draw]",
                3: "[Resign]",
                4: "[StaleMate]",
            }

        # Learn tokens
        for token in self.moves:
            # Add the token to the dictionary if it is not already there
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

        # Save the mappings to JSON files
        with open("word2idx.json", "w") as f:
            json.dump(self.word2idx, f)

        with open("idx2word.json", "w") as f:
            json.dump(self.idx2word, f)

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


if __name__ == "__main__":
    moves = pgn_extract("./dataset/1stsecond-2024.json")

    tokenizer = ChessTokenizer()
    for idx, _ in enumerate(tqdm(moves)):
        tokenizer.train(moves[idx])

    values = list(tokenizer.word2idx.values())
    if len(values) != len(set(values)):
        print("There are duplicate values in the word2idx dictionary.")
        print(f"word2idx count: {len(values)}")
        print(f"word2idx unique count: {len(set(values))}")
    else:
        print("There are no duplicate values in the word2idx dictionary.")
        print(f"word2idx count: {len(tokenizer.word2idx)}")

    values = list(tokenizer.idx2word.values())
    if len(values) != len(set(values)):
        print("There are duplicate values in the idx2word dictionary.")
        print(f"idx2word count: {len(values)}")
        print(f"idx2word unique count: {len(set(values))}")
    else:
        print("There are no duplicate values in the dictionary.")
        print(f"idx2word count: {len(tokenizer.idx2word)}")



    # text = "e4 e5 Nf3 Nc6 Bc4 Bc5 Qe2 Qf6"
    # token_ids = tokenizer.tokenize(text)
    # print(token_ids)

    # detokenized_text = tokenizer.detokenize(token_ids)
    # print(detokenized_text)
