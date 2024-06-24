'''
Build and train a transformer model
'''

import torch
from tqdm import tqdm
from transformer_blocks import GPTLanguageModel
from transformer_blocks import GPTConfig
from tokenizer import ChessTokenizer


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





with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encoder: take a string, output a list of integers
def encode(string):
    return [stoi[c] for c in string]


# decoder: take a list of integers, output a string
def decode(int_list):
    return ''.join([itos[i] for i in int_list])







# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i: i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1: i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
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
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
