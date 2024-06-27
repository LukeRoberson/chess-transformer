'''
Build and train a transformer model

Use the GPTConfig class to track hyperparameters and GPT architecture
'''

from transformer_blocks import GPTLanguageModel
from transformer_blocks import GPTConfig
from tokenizer import ChessTokenizer
from dataset import DataSet

import torch
from tqdm import tqdm


# Set up the tokenizer
tokenizer = ChessTokenizer()
tokenizer.load()
vocab_size = len(tokenizer)

# Set up the model config
config = GPTConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    tokenizer=tokenizer,
    batch_size=16,
    block_size=384,
    max_iters=500,
    eval_interval=250,
    learning_rate=3e-4,
    eval_iters=50,
    n_embd=256,
    n_head=2,
    n_layer=2,
    dropout=0.2,
    pad_token=tokenizer.pad_number,
)
print(f'using device: {config.device}')

# Dataset management
chess_dataset = DataSet(
    config,
    dataset_dir='./dataset',
)
chess_dataset.load()
chess_dataset.split(test_size=0.2)
chess_dataset.create_dataloaders()


def get_batch(split):
    data_loader = (
        chess_dataset.train_dataloader
        if split == 'train'
        else chess_dataset.test_dataloader
    )

    for x, y in data_loader:
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

# Training loop
for iter in tqdm(range(config.max_iters)):
    # every once in a while evaluate the loss on train and val sets
    if (
        (iter % config.eval_interval == 0 and iter != 0) or
        iter == config.max_iters
    ):
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f},\
            val loss {losses['val']:.4f}"
        )

    # Get a batch of training data
    xb, yb = get_batch('train')

    # Generate a mask for the input batch
    #   '[Pad]' tokens (2) are ignored in loss calculation
    mask = (xb != 2).float()

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
print(
    tokenizer.detokenize(
        m.generate(
            context,
            max_new_tokens=50
        )[0].tolist()
    )
)
