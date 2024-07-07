'''
Class for training the GPT model
This is kept separate from the model class, as the model can be used
    for training or inference
'''

from transformer_blocks import GPTConfig, GPTLanguageModel
from dataset import DataSet
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from colorama import Fore, Style


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
        weight_decay: float = 0.01,
        sched_first_cycle: int = 10,
        sched_cycle_factor: int = 1,
        sched_min_lr: float = 1e-6,
    ) -> None:
        '''
        Initialize the GPTTrainer class

        Args:
            model_config: GPTConfig
                The configuration for the model
            epochs: int
                The number of epochs to train for
            learning_rate: float
                The learning rate for the optimizer
            warmup_steps: int
                The number of warmup steps to use for the scheduler
            test_split: float
                The percentage of the dataset to use for testing
            eval_iterations: int
                The number of iterations to use for evaluation
            weight_decay: float
                The weight decay to use for the optimizer
            sched_first_cycle: int
                The number of steps in the first cycle of the scheduler
            sched_cycle_factor: int
                The factor to use for the cycle length of the scheduler
            sched_min_lr: float
                The minimum learning rate to use for the scheduler
        '''

        # Set up configuration values
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.test_split = test_split
        self.model_config = model_config
        self.eval_iterations = eval_iterations
        self.batch_size = model_config.batch_size
        self.device = model_config.device

        # Regularization
        self.weight_decay = weight_decay

        # Scheduler
        self.warmup_steps = warmup_steps
        self.sched_first_cycle = sched_first_cycle
        self.sched_cycle_factor = sched_cycle_factor
        self.sched_min_lr = sched_min_lr

    def train(
        self,
        epoch: int,
        model: GPTLanguageModel,
        dataset: DataSet,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineAnnealingWarmRestarts,
        scaler: GradScaler,
    ) -> None:
        '''
        The training loop for the GPT model

        Args:
            epoch: int
                The current epoch number
            model: GPTLanguageModel
                The model to train
            dataset: DataSet
                The dataset to train the model on
            optimizer: torch.optim.Optimizer
                The optimizer to use for training
            scheduler: CosineAnnealingWarmRestarts
                The scheduler to use for training
            scaler: GradScaler
                The scaler to use for mixed precision training
        '''

        print(f"Starting epoch #{epoch + 1} of {self.epochs}")

        # Steps (batch loop) batches within an epoch
        model.train()
        for batch_idx, batch in enumerate(
            tqdm(
                dataset.data_iter('train'),
                total=len(dataset.train_data) // self.batch_size
            )
        ):
            optimizer.zero_grad(set_to_none=True)

            # Scheduler Warmup Phase
            if epoch < self.warmup_steps:
                lr = self.learning_rate * (epoch / self.warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                # After warmup, adjust learning rate based on scheduler
                scheduler.step(epoch - self.warmup_steps)

            # Move to GPU
            xb, yb = batch
            xb, yb = xb.to(self.device), yb.to(self.device)

            # Generate a mask for the input batch
            #   '[Pad]' tokens (2) are ignored in loss calculation
            mask = (xb != 2).float()

            # Forward pass
            with autocast():
                logits, loss = model(xb, yb)

            # Mixed precision backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update the scheduler
            scheduler.step(epoch + batch_idx / len(dataset.train_data))

            # Free up VRAM
            del xb, yb, mask, logits, loss
            torch.cuda.empty_cache()

        # Evaluate every full epoch (epoch's are large)
        losses = self.estimate_loss(
            dataset=dataset,
            model=model,
        )
        print(
            Fore.GREEN,
            f"Epoch #{epoch + 1} results: "
            f"training loss {losses['train']:.4f}, "
            f"validation loss {losses['val']:.4f}",
            Style.RESET_ALL
        )

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
