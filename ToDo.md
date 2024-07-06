# To Do

## Front End
* Hyperparameters for training the transformer
    * GUI done, code needs to be added
* Open folder and open file windows to open on top
* Add a training progress tqdm to the tokenizer

## Transformer
* Move the estimate_loss() function to the GPT model class
* Move the training loop into the GPT model class
* Save/load the model
* Implement a scaler such as amp.GradScaler()

## Generation
* Put move numbers back in when generating
* Stop generating when we reach an end token
* Add temperature

## Dataset
* Add an optional max game size option
* Consider reshuffling the dataset with each epoch

# Testing
* See if a different activation function in the FFN has an effect
* See if adding biases in the attention heads makes a difference
* Can we dynamically adjust batch size based on consumed VRAM?
