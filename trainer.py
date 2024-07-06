'''
Class for training the GPT model
This is kept separate from the model class, as the model can be used
    for training or inference
'''

from transformer_blocks import GPTConfig


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
