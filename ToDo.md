# To Do

## Front End
* Hyperparameters for training the transformer
    * GUI done, code needs to be added
* Open folder and open file windows to open on top
* Add a training progress tqdm to the tokenizer
* Resume training option
* Checkpoint save filename

## Transformer
* Save model architecture (to handle dynamic block sizes based on game limit)
* Append epoch number to model filenames

## Generation
* Put move numbers back in when generating
* Stop generating when we reach an end token
* Add temperature

# Tokenizer
* Add support for parallel processing when tokenizing strings
* Can games be tokenized in batches?

# Testing
* See if a different activation function in the FFN has an effect
* See if adding biases in the attention heads makes a difference

# Wishlist
* See if it's possible to include JSON data in the checkpoint (for tokenizer)
* Separate forward pass from the loss?
* Can we use data augmentation to improve the dataset?
* During finetuning, can we add annotations to explain moves?
