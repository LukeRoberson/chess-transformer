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
            "0": "[Open]",
            "1": "[CheckMate]",
            "2": "[Draw]",
            "3": "[Resign]",
            "4": "[StaleMate]",
        }

        self.idx2word = {
            "[Open]": "0",
            "[CheckMate]": "1",
            "[Draw]": "2",
            "[Resign]": "3",
            "[StaleMate]": "4",
        }

        # Learn new tokens
        forward = {move: idx for idx, move in enumerate(self.moves)}
        self.word2idx.update(forward)
        reverse = {idx: move for idx, move in enumerate(self.moves)}
        self.idx2word.update(reverse)

        # Save the mappings to JSON files
        with open("word2idx.json", "w") as f:
            json.dump(self.word2idx, f)

        with open("idx2word.json", "w") as f:
            json.dump(self.idx2word, f)

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
            self.word2idx[token] for
            token in tokens if
            token in self.word2idx
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
            self.idx2word[token_id] for
            token_id in token_ids if
            token_id in self.idx2word
        ]
        text = " ".join(tokens)

        return text


if __name__ == "__main__":
    # Example usage
    moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "Qe2", "Qf6"]
    tokenizer = ChessTokenizer()
    tokenizer.train(moves)

    text = "e4 e5 Nf3 Nc6 Bc4 Bc5 Qe2 Qf6"
    token_ids = tokenizer.tokenize(text)
    print(token_ids)  # Output: [0, 1, 2, 3, 4, 5, 6, 7]

    detokenized_text = tokenizer.detokenize(token_ids)
    print(detokenized_text)  # Output: "e4 e5 Nf3 Nc6 Bc4 Bc5 Qe2 Qf6"
