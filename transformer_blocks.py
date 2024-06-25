'''
Building blocks of the transformer model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)

        # compute attention scores ("affinities")
        #   (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    config,
                    head_size
                )
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(head_size * config.n_head, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(
            config,
            head_size,
        )
        self.ffwd = FeedFoward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        config,
        vocab_size
    ):
        super().__init__()
        self.device = config.device
        self.block_size = config.block_size
        self.pad_token = config.pad_token

        # An embedding layer is a 2D tensor
        #   Each row is a vector that represents a token
        #   The embedding table is trained with each forward pass and backprop
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            config.n_embd
        )

        self.position_embedding_table = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        self.blocks = nn.Sequential(
            *[
                Block(config)
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        # better init, not covered in the original GPT video,
        #   but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        '''
        The forward pass of the model
            When the model is called, the forward pass is executed
            This could be for training or inference
            No targets are passed for inference

        This method works heavily with tensors. Three dimensions are used:
            B: Batch size
            T: Sequence length (AKA 'time')
            C: Number of channels (or dimensions in the embedding)

        Loss is calculated with the cross-entropy loss function
            This ignores the 'pad' token when learning

        Args:
            idx: torch.Tensor
                The input tensor
                The indices of the tokens in the context

            targets: torch.Tensor
                The target tensor
                The indices of the tokens in the context
        '''

        # Get the batch size and sequence length as B, T
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        ''' Learn more about this '''
        tok_emb = self.token_embedding_table(idx.long())  # (B,T,C)

        ''' Learn more about this '''
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)

        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T).long()

            # Calculate the loss
            #   The ignore_index is a special token that should be ignored
            #   ([Pad] token in our case)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=self.pad_token
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class GPTConfig():
    '''
    Configuration for GPT model
    Stores model hyperparameters
        This means less passing of arguments to functions
    '''

    def __init__(
        self,
        device,
        tokenizer,
        batch_size=64,
        block_size=256,
        max_iters=5000,
        eval_interval=500,
        learning_rate=3e-4,
        eval_iters=200,
        n_embd=384,
        n_head=4,
        n_layer=4,
        dropout=0.2,
        pad_token=0,
    ):
        '''
        Setup the hyperparameters for the model

        Args:
            device: torch.device
                'cuda' or 'cpu'

            tokenizer: ChessTokenizer
                Tokenizer for the model

            batch_size: int
                The size of the batch sent to the model
                Choose the right size for available GPU VRAM

            block_size: int
                The transformers context length

            max_iters: int
                The number of epochs to train for

            eval_interval: int
                How often to evaluate the model
                Runs the loss on the validation set

            learning_rate: float
                The learning rate for the model

            eval_iters: int
                Context length for evaluation

            n_embd: int
                Embedding size
                This is the dimension of the embedding vectors

            n_head: int
                Number of heads for self-attention

            n_layer: int
                Number of transformer blocks (decoder layers)

            dropout: float
                Dropout rate for the model
                A type of regularization

            pad_token: int
                The token for padding
                This is the token number used for padding sequences
        '''

        self.device = device
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.block_size = block_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.eval_iters = eval_iters
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout

        self.pad_token = pad_token
