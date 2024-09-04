'''
Class for training the GPT model
This is kept separate from the model class, as the model can be used
    for training or inference
'''

from transformer_blocks import GPTConfig, GPTLanguageModel
from dataset import ManageDataSet
from tokenizer import ChessTokenizer

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
from torch.cuda import Stream

from tqdm import tqdm
from colorama import Fore, Style
import math
import time


class GPTTrainer():
    '''
    Class for training the GPT model

    Methods:
        train:
            Train the model
            Includes resuming from a checkpoint
        estimate_loss:
            Estimate the loss of the model
            Calculates training and validation loss
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
        model: GPTLanguageModel,
        dataset: ManageDataSet,
        optimizer: torch.optim.Optimizer,
        scheduler: CosineAnnealingWarmRestarts,
        scaler: GradScaler,
        tokenizer: ChessTokenizer,
        percent: float = 1.0,
        resume: bool = False,
        checkpoint: str = 'model.pth',
    ) -> None:
        '''
        The training loop for the GPT model

        Supports resuming training from a checkpoint
            To resume, set resume=True and provide the checkpoint path
            This will pick up at the next epoch

        Chunking is supported by setting the percentage of the dataset to use
            This is useful for large datasets where the entire dataset
                cannot be loaded at once

        Args:
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
            percent: float
                The size of the dataset to use in each chunk
            resume: bool
                Whether to resume training from a checkpoint
            checkpoint: str
                The path to the checkpoint to resume from
        '''

        # Setup CUDA streams for faster data transfer
        data_stream = Stream()
        comp_stream = Stream()

        # Set the starting epoch
        epoch = 0

        # Initialise a losses dictionary
        loss_history = {}

        # Resume training from a checkpoint, will update starting epoch
        if resume:
            # Load the model from the checkpoint
            epoch, loss_history = model.load_checkpoint(
                optimizer=optimizer,
                scheduler=scheduler,
                filename=checkpoint,
            )

            # Print previous loss history
            for epoch_num, losses in loss_history.items():
                print(
                    Fore.GREEN,
                    f"Epoch #{epoch_num + 1} results: "
                    f"training loss {losses['train']:.4f}, "
                    f"validation loss {losses['val']:.4f}",
                    Style.RESET_ALL
                )

            # The next epoch is the completed epoch + 1
            epoch += 1
            print(f"Resuming training from epoch {epoch + 1}...")

        if epoch >= self.epochs:
            print("Training complete!")
            return

        # Start loading the data in a separate thread
        dataset.start_data_loading(percentage=percent)

        # The main training loop
        for epoch_num in range(epoch, self.epochs):
            epoch_start_time = time.time()
            print(f"\n\nEpoch {epoch_num + 1} of {self.epochs}")

            # Chunking - Loop over chunks per epoch
            for chunk in range(math.ceil(1.0 / percent)):
                chunk_start_time = time.time()

                # Fetch the next chunk from the queue
                train_dataloader, train_data_size = (
                    dataset.data_queue.get()
                )

                # Steps (batch loop) batches within an epoch
                model.train()
                print(
                    Fore.YELLOW,
                    f"\nChunk {chunk + 1} of {math.ceil(1.0 / percent)}",
                    Style.RESET_ALL
                )
                for batch_idx, batch in enumerate(
                    tqdm(
                        train_dataloader,
                        total=train_data_size // self.batch_size,
                        colour='yellow',
                        desc="Training batches",
                        leave=False,
                    )
                ):
                    optimizer.zero_grad(set_to_none=True)

                    # Scheduler Warmup Phase
                    if epoch_num < self.warmup_steps:
                        lr = self.learning_rate * (
                            epoch_num / self.warmup_steps
                        )
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        # After warmup, adjust learning rate based on scheduler
                        scheduler.step(epoch_num - self.warmup_steps)

                    # Move to GPU, using CUDA streams for faster data transfer
                    xb, yb = batch
                    with torch.cuda.stream(data_stream):
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)

                    # Forward pass and backpropagation, using CUDA streams
                    with torch.cuda.stream(comp_stream):
                        # Forward pass
                        with autocast('cuda'):
                            _, loss = model(xb, yb)

                        # Mixed precision backpropagation
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        # Update the scheduler
                        scheduler.step(
                            epoch + batch_idx / dataset.train_data_size
                        )

                print(
                    Fore.YELLOW,
                    f" Training time: {time.time() - chunk_start_time:.1f}s",
                    Style.RESET_ALL
                )

            # Create an evaluation dataset
            dataset.get_test_dataset()

            # Evaluate the model on the test dataset
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

            # Store the losses in the history every epoch
            loss_history[epoch_num] = losses

            # Save the model
            model.save_checkpoint(
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch_num,
                loss_history=loss_history,
            )

            # Print the time taken for the epoch
            seconds = time.time() - epoch_start_time
            minutes = seconds // 60
            seconds = seconds % 60
            print(
                Fore.YELLOW,
                f"Epoch {epoch_num + 1} duration: ",
                f"{minutes:.0f}:{seconds:.0f}",
                Style.RESET_ALL
            )

            # Generate a sample from the model
            sequence = model.generate(
                context=None,
                max_new_tokens=50
            )[0].tolist()
            print(tokenizer.detokenize(sequence))

        # End of loop, stop the thread
        dataset.stop_data_loading()

    @torch.no_grad()
    def estimate_loss(
        self,
        dataset: ManageDataSet,
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
            for batch_index in tqdm(
                range(self.eval_iterations),
                desc=f"Estimating {split} loss",
                colour='green',
                leave=False,
            ):
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
