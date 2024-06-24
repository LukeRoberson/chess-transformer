'''
Build and train a transformer model
'''

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformer_blocks import GPTLanguageModel
from transformer_blocks import GPTConfig
from tokenizer import ChessTokenizer
from sklearn.model_selection import train_test_split
import os
import re
import json


# Set up the model config
config = GPTConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    batch_size=256,
    block_size=256,
    max_iters=500,
    eval_interval=250,
    learning_rate=3e-4,
    eval_iters=200,
    n_embd=256,
    n_head=2,
    n_layer=2,
    dropout=0.2
)
print(f'using device: {config.device}')

# Set up the tokenizer
tokenizer = ChessTokenizer()
tokenizer.load()
vocab_size = len(tokenizer)

# Dataset management
dataset_dir = './dataset'
game_list = []

for file in tqdm(os.listdir(dataset_dir), desc='Finding JSON files'):
    if file.endswith('.json'):
        try:
            with open(f"{dataset_dir}/{file}", "r") as f:
                moves = json.load(f)

            # Parse the JSON file for games
            year = list(moves.keys())[0]
            for month in moves[year]:
                for game in moves[year][month]:
                    game = re.sub(r"\d{1,3}\. ", "", game['pgn']).strip()
                    game_list.append(tokenizer.tokenize(game))

        except (IOError, json.JSONDecodeError) as e:
            print(f"Error processing file {file}: {e}")

# Get input and target sequences
#   Target is input shifted right by one token
input_sequences = []
target_sequences = []

for game in game_list:
    input_sequences.append(game[:-1])
    target_sequences.append(game[1:])

# Split input_sequences and target_sequences into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    input_sequences,
    target_sequences,
    test_size=0.2,
    random_state=42
)

# Create tensor datasets
train_dataset = TensorDataset(
     torch.Tensor(train_data),
     torch.Tensor(train_labels)
)
test_dataset = TensorDataset(
     torch.Tensor(test_data),
     torch.Tensor(test_labels)
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4)
test_loader = DataLoader(test_dataset)


def get_batch(split):
    dataloader = train_loader if split == 'train' else test_loader
    data_iter = iter(dataloader)
    x, y = next(data_iter)
    x, y = x.to(config.device), y.to(config.device)
    return x, y


# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
#     x = torch.stack([data[i: i + config.block_size] for i in ix])
#     y = torch.stack([data[i + 1: i + config.block_size + 1] for i in ix])
#     x, y = x.to(config.device), y.to(config.device)
#     return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = GPTLanguageModel(config, vocab_size)
m = model.to(config.device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

for iter in tqdm(range(config.max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f},\
            val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
