# To Do

## Front End
* Hyperparameters for training thr transformer
    * Maybe add profiles for common architectures
* Open folder and open file windows to open on top
* Add a training progress tqdm to the tokenizer

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
