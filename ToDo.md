# To Do

## Front End
* Hyperparameters for training the transformer
    * GUI done, code needs to be added
* Open folder and open file windows to open on top
* Add a training progress tqdm to the tokenizer

## Tokenizer
* Found some tokens with an '@' symbol (unsure why?)

## Transformer
<<<<<<< HEAD
* Resume training
* Save losses in the model checkpoint
* Save model architecture
=======
* Save/load the model
>>>>>>> 509b1a4d637fef28d7fedbe0525b8430c0b0e0b1

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
