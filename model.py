'''
Build and train a transformer model

Use the GPTConfig class to track hyperparameters and GPT architecture
Various classes are used from the transformer_blocks module
'''

from transformer_blocks import GPTLanguageModel
from transformer_blocks import GPTConfig
from tokenizer import ChessTokenizer
from dataset import DataSet

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import sys
from colorama import Fore, Style


# Set up the tokenizer
tokenizer = ChessTokenizer()
tokenizer.load()

# Set up the model config
config = GPTConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    tokenizer=tokenizer,
    batch_size=32,
    block_size=384,
    epochs=2,
    learning_rate=2e-4,
    eval_iters=50,
    n_embd=256,
    n_head=2,
    n_layer=2,
    dropout=0.2,
    pad_token=tokenizer.pad_number,
    test_split=0.2,
    warmup_steps=10,
)
print(f'using device: {config.device}')

# Dataset management
chess_dataset = DataSet(
    config,
    dataset_dir='./dataset',
)
chess_dataset.load()
chess_dataset.split(test_size=config.test_split)
chess_dataset.create_dataloaders()

# Create the model
model = GPTLanguageModel(
    config=config,
    vocab_size=len(tokenizer),
).to(config.device)
print(f'{model.param_count/1e6}M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)

# Initialize the scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=1,
    eta_min=1e-6
)

# Training loop (epoch loop, full dataset)
try:
    for epoch in range(config.epochs):
        print(f"Starting epoch #{epoch + 1} of {config.epochs}")

        # Steps (batch loop) batches within an epoch
        model.train()
        for batch_idx, batch in enumerate(
            tqdm(
                chess_dataset.data_iter('train'),
                total=len(chess_dataset.train_data) // config.batch_size
            )
        ):
            optimizer.zero_grad(set_to_none=True)

            # Scheduler Warmup Phase
            if epoch < config.warmup_steps:
                lr = config.learning_rate * (epoch / config.warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                # After warmup, adjust learning rate based on scheduler
                scheduler.step(epoch - config.warmup_steps)

            # Move to GPU
            xb, yb = batch
            xb, yb = xb.to(config.device), yb.to(config.device)

            # Generate a mask for the input batch
            #   '[Pad]' tokens (2) are ignored in loss calculation
            mask = (xb != 2).float()

            # Forward pass
            logits, loss = model(xb, yb)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / len(chess_dataset.train_data))

            # Free up memory
            del xb, yb, mask, logits, loss
            torch.cuda.empty_cache()

        # Evaluate every full epoch (epoch's are large)
        losses = model.estimate_loss(chess_dataset)
        print(
            Fore.GREEN,
            f"Epoch #{epoch + 1} results: "
            f"training loss {losses['train']:.4f}, "
            f"validation loss {losses['val']:.4f}",
            Style.RESET_ALL
        )

except KeyboardInterrupt:
    print('Interrupted by user. Exiting...')
    sys.exit(0)

# Generate a sequence of tokens from scratch
sequence = model.generate(context=None, max_new_tokens=50)[0].tolist()

# Detokenize the sequence and print it
print(tokenizer.detokenize(sequence))
