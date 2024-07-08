# To Do

## Front End
* Hyperparameters for training the transformer
    * GUI done, code needs to be added
* Open folder and open file windows to open on top
* Add a training progress tqdm to the tokenizer
* Resume training option
* Checkpoint save filename

## Tokenizer
* Found some tokens with an '@' symbol (unsure why?)

## Transformer
* Save losses in the model checkpoint
* When resuming, print losses so far
* Save model architecture (to handle dynamic block sizes based on game limit)
* Append epoch number to model filenames

## Generation
* Put move numbers back in when generating
* Stop generating when we reach an end token
* Add temperature

## Dataset
* Train on a percentage of the dataset per epoch
    * Recreate the dataset with random files each time

# Testing
* See if a different activation function in the FFN has an effect
* See if adding biases in the attention heads makes a difference

# Wishlist
* See if it's possible to include JSON data in the checkpoint (for tokenizer)
