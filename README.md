# Chess Transformer

A transformer model designed to learn a player's style of playing, with the intent to play like that player.

## Web Frontend

A front-end has been provided using Gradio, to make usage easier.

Run web.py, which will start a local web server. Go to the URL provided to access the front-end.

There are three tabs provided:
* Tokenizer: Used to train the tokenizer
* Trainer: Used to train the model
* Generator: Used to generate chess moves

## Dataset

The dataset to train the tokenizer and the transformer needs to consist of JSON files in a particular format:
```
{
    YEAR: {
        "01": [],
        "02": [],
        ...
        "12": [],
    }
}
```

Each JSON file represents a particular player's games for an entire year. There is only one year per JSON file, and months nested under the year. NOTE: Not all months may be present.

Each month has a list of games. Each of these is formatted like this:
```
{
    "pgn": "1. d4 d5 2. Nf3 Nf6 3. g3 c6 4. Bg2 Bg4",
    "result": "1/2-1/2",
    "1": ["d4", "d5"],
    "2": ["Nf3", "Nf6"],
    ...
    "n": ["Bg2", "Bg4"]
},
```

* The "pgn" part is the raw list of moves in standard PGN format, including move number.
* The "result" is how the game ended, in standard notation (the example above shows a draw)
* There are 'n' number of moves listed after this. This breaks the PGN into move pairs

## Tokenizer

The tokenizer is a class called ChessTokenizer. No arguments are required to initialise it.

### Word Based

A custom tokenizer is required, as we need word-based tokens. Each chess move (in standard chess notation) is like a word in a sentence.

Sub-word and character based tokenizers will invent new words. This is not suitable, as there are specific chess moves that are valid, and everything else is invalid.

### Training

Train the tokenizer by calling the train() method, and passing a string of moves. Pass a list of files with 'file_list='. This is a list of JSON files that make up the dataset to train the tokenizer on.

The trainer will automatically save the results to two JSON files.
* word2idx.json
* idx2word.json

Optionally pass a directory as a string to 'save_path=' to change the location where these files are saved. By default they're saved in the working directory.

### Resume Training

While training, a 'resume.txt' file is created. This is a list of all files that the tokenizer has been trained on. By default, this is updated with every 50 training files. This can be changed by setting the CHECKPOINT constant at the top of the tokenizer.py file.

If training is interrupted (which can easily happen over the course of 80,000 files in the dataset), you can use the resume.txt file to skip files in the dataset that have already been used in the training process.

### Saving and Loading

The json_save() method can be used to save a tokenizer to the two JSON files. This is implicitly called during training, and usually wouldn't need to be called manually.

Call the load() method to load the JSON files back into the object. This makes the ChessTokenizer.word2idx and ChessTokenizer.idx2word objects available for tokenization and detokenization.

### Tokenizing and Detokenizing

The tokenize() method will convert a string into a list tokens.

The detokenize() method will convert from a list of tokens to a string.

## Transformer

The transformer is primarily implemented in two files:
* transformer_blocks.py - The classes used to build the transformer
* model.py - The code to build the model architecture, load tokenizer, load dataset, and train the model

### Creating a Model

1. Load a tokenizer
2. Create model configuration using the GPTConfig class
3. Prepare a dataset 
4. Create the model, optimizer, etc

### Dataset Preparation

The dataset.py file contains the DataSet() class. This loads and prepares the dataset into dataloaders that the model can use.

1. Create an instance of the DataSet class, passing the GPTConfig object, and optionally specifying the location of the dataset ('./dataset' is used by default)
2. Run the load() method to read the JSON files, and extract the games
3. Run the split() method to split the dataset into train and test sets. Optionally pass the 'test_size=' parameter to define the split (0.2, or 20%, is the default)
4. Run the create_dataloaders() method to load the dataset in to standardised PyTorch dataloader structures

There are two dataloaders that can be accessed from the class:
* train_dataloader
* test_dataloader
