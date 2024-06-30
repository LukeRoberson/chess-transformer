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
from tqdm import tqdm
import sys


# Set up the tokenizer
tokenizer = ChessTokenizer()
tokenizer.load()

# Set up the model config
config = GPTConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    tokenizer=tokenizer,
    batch_size=256,
    block_size=384,
    max_iters=200,
    eval_interval=100,
    learning_rate=3e-4,
    eval_iters=50,
    n_embd=256,
    n_head=2,
    n_layer=2,
    dropout=0.2,
    pad_token=tokenizer.pad_number,
    test_split=0.2,
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

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)

# Training loop
try:
    for iter in tqdm(range(config.max_iters)):
        # Get a batch of training data
        xb, yb = chess_dataset.get_batch('train')

        # Generate a mask for the input batch
        #   '[Pad]' tokens (2) are ignored in loss calculation
        mask = (xb != 2).float()

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if (
            (iter % config.eval_interval == 0 and iter != 0) or
            iter == config.max_iters
        ):
            losses = model.estimate_loss(chess_dataset)
            print(
                f"step {iter}: train loss {losses['train']:.4f},\
                val loss {losses['val']:.4f}"
            )

except KeyboardInterrupt:
    print('Interrupted by user. Exiting...')
    sys.exit(0)

# Generate a sequence of tokens based on the context
#   The context is nothing at this point
sequence = model.generate(context=None, max_new_tokens=50)[0].tolist()

# Detokenize the sequence and print it
print(tokenizer.detokenize(sequence))
