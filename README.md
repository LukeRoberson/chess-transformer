# Chess Transformer

A transformer model designed to learn a player's style of playing, with the intent to play like that player.

## Tokenizer

A custom tokenizer is required, as we need word-based tokens. Each chess move (in standard chess notation) is like a word in a sentence.

Sub-word and character based tokenizers will invent new words. This is not suitable, as there are specific chess moves that are valid, and everything else is invalid.

Train the tokenizer by calling the train() method, and passing a string of moves. The trainer will automatically save the results to two JSON files.

Call the load() method to load the JSON files back into the object

The tokenize() method will convert a string into a list tokens, and the detokenize() method will convert from a list of tokens to a string.
