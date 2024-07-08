'''
Build and train a transformer model

Use the GPTConfig class to track hyperparameters and GPT architecture
Various classes are used from the transformer_blocks module
'''

from transformer_blocks import GPTLanguageModel
from transformer_blocks import GPTConfig
from trainer import GPTTrainer
from tokenizer import ChessTokenizer
from dataset import DataSet

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler

import sys


# Set up the tokenizer
tokenizer = ChessTokenizer()
tokenizer.load()

# Set up the model config
model_config = GPTConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    tokenizer=tokenizer,
    batch_size=64,
    block_size=384,
    n_embd=256,
    n_head=2,
    n_layer=2,
    dropout=0.2,
    pad_token=tokenizer.pad_number,
)
print(f'using device: {model_config.device}')

# Set up the GPTTrainer
trainer = GPTTrainer(
    epochs=10,
    learning_rate=2e-4,
    warmup_steps=10,
    test_split=0.2,
    model_config=model_config,
    eval_iterations=50,
    weight_decay=0.01,
    sched_first_cycle=10,
    sched_cycle_factor=2,
    sched_min_lr=1e-6,
)

# Dataset management
chess_dataset = DataSet(
    model_config=model_config,
    train_config=trainer,
    dataset_dir='./dataset',
)
block_size = chess_dataset.load(
    min_moves=6,
    max_moves=190,
)
chess_dataset.split(
    test_size=trainer.test_split
)
chess_dataset.create_dataloaders()

# Update the block size in the model config
#   No sense being larger than the dataset block size
model_config.block_size = block_size

# Create the model
model = GPTLanguageModel(
    config=model_config,
    vocab_size=len(tokenizer),
).to(model_config.device)
print(f'{model.param_count/1e6}M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=trainer.learning_rate,
    weight_decay=trainer.weight_decay,
)

# Initialize the scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=trainer.sched_first_cycle,
    T_mult=trainer.sched_cycle_factor,
    eta_min=trainer.sched_min_lr,
)

# Initialise the scaler
scaler = GradScaler()

# Training loop (epoch loop, full dataset)
try:
    for epoch in range(trainer.epochs):
        trainer.train(
            epoch=epoch,
            model=model,
            dataset=chess_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

except KeyboardInterrupt:
    print('Interrupted by user. Exiting...')
    sys.exit(0)

# Generate a sequence of tokens from scratch
sequence = model.generate(context=None, max_new_tokens=50)[0].tolist()

# Detokenize the sequence and print it
print(tokenizer.detokenize(sequence))
