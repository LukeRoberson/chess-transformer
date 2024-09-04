'''
Build and train a transformer model

Use the GPTConfig class to track hyperparameters and GPT architecture
Various classes are used from the transformer_blocks module
'''

from transformer_blocks import GPTLanguageModel
from transformer_blocks import GPTConfig
from trainer import GPTTrainer
from tokenizer import ChessTokenizer
from dataset import ManageDataSet
from config import Config

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler
from colorama import Fore, Style


# Read the configuration file
settings = Config(config_file='config.yaml')


# Set up the tokenizer
tokenizer = ChessTokenizer()
tokenizer.load()

# Set up the model config
model_config = GPTConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    tokenizer=tokenizer,
    batch_size=settings.dataset['batch_size'],
    block_size=settings.model['block_size'],
    n_embd=settings.model['embedding_size'],
    n_head=settings.model['heads'],
    n_layer=settings.model['layers'],
    dropout=settings.regularization['dropout'],
    pad_token=tokenizer.pad_number,
)

# Set up the GPTTrainer
trainer = GPTTrainer(
    epochs=settings.training['epochs'],
    learning_rate=settings.training['learning_rate'],
    warmup_steps=settings.training['warmup_steps'],
    test_split=settings.dataset['test_split'],
    model_config=model_config,
    eval_iterations=settings.training['eval_iterations'],
    weight_decay=settings.regularization['weight_decay'],
    sched_first_cycle=settings.scheduler['first_cycle'],
    sched_cycle_factor=settings.scheduler['cycle_factor'],
    sched_min_lr=settings.scheduler['min_lr'],
)

# Dataset management
chess_dataset = ManageDataSet(
    model_config=model_config,
    dataset_dir=settings.dataset['path'],
)

# Create the model
model = GPTLanguageModel(
    config=model_config,
    vocab_size=len(tokenizer),
).to(model_config.device)


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
scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

# Print the model configuration
print(
    Fore.YELLOW, "----------------------------------------------------"
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Parameter",
    Fore.YELLOW, "        | ",
    Fore.GREEN, "Value"
)
print(
    Fore.YELLOW, "----------------------------------------------------"
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Device",
    Fore.YELLOW, "           | ",
    Fore.GREEN, model_config.device
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Batch size",
    Fore.YELLOW, "       | ",
    Fore.GREEN, model_config.batch_size
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Block size",
    Fore.YELLOW, "       | ",
    Fore.GREEN, model_config.block_size
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Embedding size",
    Fore.YELLOW, "   | ",
    Fore.GREEN, model_config.n_embd
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Heads",
    Fore.YELLOW, "            | ",
    Fore.GREEN, model_config.n_head
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Layers",
    Fore.YELLOW, "           | ",
    Fore.GREEN, model_config.n_layer
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Dropout",
    Fore.YELLOW, "          | ",
    Fore.GREEN, model_config.dropout
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Weight decay",
    Fore.YELLOW, "     | ",
    Fore.GREEN, trainer.weight_decay
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Learning rate",
    Fore.YELLOW, "    | ",
    Fore.GREEN, trainer.learning_rate
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Warmup steps",
    Fore.YELLOW, "     | ",
    Fore.GREEN, trainer.warmup_steps
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "First cycle",
    Fore.YELLOW, "      | ",
    Fore.GREEN, trainer.sched_first_cycle
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Cycle factor",
    Fore.YELLOW, "     | ",
    Fore.GREEN, trainer.sched_cycle_factor
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Min LR",
    Fore.YELLOW, "           | ",
    Fore.GREEN, trainer.sched_min_lr
)
print(
    Fore.YELLOW, "| ",
    Fore.GREEN, "Total Parameters",
    Fore.YELLOW, " | ",
    Fore.GREEN, f'{model.param_count/1e6}M parameters'
)
print(
    Fore.YELLOW, "----------------------------------------------------")
print(Style.RESET_ALL)


# Training loop (epoch loop, full dataset)
trainer.train(
    model=model,
    dataset=chess_dataset,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    tokenizer=tokenizer,
    resume=settings.training['resume'],
    percent=settings.dataset['chunk_percent'],
    checkpoint=settings.training['checkpoint'],
)

print(Fore.GREEN, "Training complete")
print(Style.RESET_ALL)
