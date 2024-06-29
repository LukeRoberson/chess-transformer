# To Do

## Tokenizer
* Specify a path to load the tokenizer
* Decode: Remove PAD token
* Consider whether we need '#' (Check) in tokens
    * This could be added externally when moved are validated
* Consider adding different types of end tokens
    * These can match the way the game ends
* Update load() to allow different paths and filenames
* Load(): Handle errors if the file does not exist

## Front End
* Path to load the tokenizer
    * Have added basic GUI
    * Move resume code to the training function
    * Needs testing
* Hyperparameters for training
* Maybe add profiles for common architectures
* Open folder and open file windows to open on top

## Transformer
* Add padding to datasets so they are the same length
* Add Ctrl-C handling
* Move the estimate_loss() function to the GPT model class
* Move the training loop into the GPT model class
* Add a scheduler

## Generation
* Put move numbers back in when generating
* Stop generating when we reach an end token
* Add temperature

## Dataset
* Add an optional max game size option
* Move the get_batch() function to the DataSet class
