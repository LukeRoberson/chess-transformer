'''
The tokenizer
This collects a list of known chess moves in standard chess notation

This needs to be word-based, not character-based or sub-word
    This is because we can not change the chess moves or invent new ones
'''

from typing import List
import json


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

        self.moves = moves

        # Create dictionaries, an add special tokens
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

        # Learn new tokens
        forward = {move: idx for idx, move in enumerate(self.moves, start=5)}
        self.word2idx.update(forward)
        reverse = {idx: move for idx, move in enumerate(self.moves, start=5)}
        self.idx2word.update(reverse)

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


if __name__ == "__main__":
    # Example usage
    moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "Qe2", "Qf6"]
    tokenizer = ChessTokenizer()
    tokenizer.load()

    text = "e4 e5 Nf3 Nc6 Bc4 Bc5 Qe2 Qf6"
    token_ids = tokenizer.tokenize(text)
    print(token_ids)

    detokenized_text = tokenizer.detokenize(token_ids)
    print(detokenized_text)
