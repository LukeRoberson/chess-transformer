'''
Class for training the GPT model
This is kept separate from the model class, as the model can be used
    for training or inference
'''

from transformer_blocks import GPTConfig, GPTLanguageModel
from dataset import DataSet
import torch


class GPTTrainer():
    '''
    Class for training the GPT model
    '''

    def __init__(
        self,
        model_config: GPTConfig,
        epochs: int = 2,
        learning_rate: float = 2e-4,
        warmup_steps: int = 10,
        test_split: float = 0.2,
        eval_iterations: int = 50,
    ) -> None:
        '''
        Initialize the GPTTrainer class
        '''

        # Set up configuration values
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.test_split = test_split
        self.model_config = model_config
        self.eval_iterations = eval_iterations

    @torch.no_grad()
    def estimate_loss(
        self,
        dataset: DataSet,
        model: GPTLanguageModel,
    ) -> dict:
        '''
        Estimate the loss of the model
        Note, training is disabled during this process using
            no_grad() and eval()

        Args:
            dataset: DataSet
                The dataset to evaluate the model on

        Returns:
            dict
                A dictionary of the loss on the training and validation sets
        '''

        # Dictionary to store the average losses
        average_losses_train = {}

        # Disable training
        model.eval()

        for split in ['train', 'val']:
            # Initialize the losses tensor to all zeros
            losses = torch.zeros(self.eval_iterations)

            # Loop through the evaluation iterations
            for batch_index in range(self.eval_iterations):
                # Get a batch of data
                X, Y = dataset.get_batch(split)

                # Run the forward pass and get the loss
                _, loss = model(X, Y)

                # Store the loss in the tensor
                losses[batch_index] = loss.item()

            average_losses_train[split] = losses.mean()

        # Enable training again
        model.train()

        return average_losses_train
