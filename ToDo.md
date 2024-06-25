# To Do

## Tokenizer
* Specify a path to load the tokenizer
* Decode: Remove PAD token
* Consider whether we need '#' (Check) in tokens
    * This could be added externally when moved are validated
* Consider adding different types of end tokens
    * These can match the way the game ends

## Front End
* Path to load the tokenizer
* Hyperparameters for training
* Maybe add profiles for common architectures

## Transformer
* Add padding to datasets so they are the same length
* Transformer to ignore "[Pad]" during training
* Add Ctrl-C handling

## Dataset
* Add an optional max game size option
* Move the get_batch() function to the DataSet class
