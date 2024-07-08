'''
Use the model to generate chess moves
'''

from transformer_blocks import GPTLanguageModel
from transformer_blocks import GPTConfig
from tokenizer import ChessTokenizer
import torch


# Set up the tokenizer
tokenizer = ChessTokenizer()
tokenizer.load()

# Set up the model config
model_config = GPTConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    tokenizer=tokenizer,
    batch_size=64,
    block_size=192,
    n_embd=256,
    n_head=2,
    n_layer=2,
    dropout=0.2,
    pad_token=tokenizer.pad_number,
)
print(f'using device: {model_config.device}')

# Create the model
model = GPTLanguageModel(
    config=model_config,
    vocab_size=len(tokenizer),
).to(model_config.device)
print(f'{model.param_count/1e6}M parameters')

# Load the model
model.load_checkpoint()

# Generate a sequence of tokens from scratch
sequence = model.generate(context=None, max_new_tokens=50)[0].tolist()

# Detokenize the sequence and print it
print(tokenizer.detokenize(sequence))
