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
print(f'using device: {model_config.device}')

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
scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

# Training loop (epoch loop, full dataset)
trainer.train(
    model=model,
    dataset=chess_dataset,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    resume=settings.training['resume'],
    percent=settings.dataset['chunk_percent'],
    checkpoint=settings.training['checkpoint'],
)

# Generate a sequence of tokens from scratch
sequence = model.generate(context=None, max_new_tokens=50)[0].tolist()

# Detokenize the sequence and print it
print(tokenizer.detokenize(sequence))
