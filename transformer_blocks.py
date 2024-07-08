'''
Building blocks of the transformer model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import ChessTokenizer

from typing import Optional, Tuple


class Head(nn.Module):
    '''
    A single head of self-attention
    Mutiple heads will be created in parallel to learn different relationships
        between tokens

    Methods:
        __init__:
            Constructor method for the Head class
            This defines the layers and structures used in the head
        forward:
            The forward pass of the head
    '''

    def __init__(
        self,
        config: 'GPTConfig',
        head_size: int
    ) -> None:
        '''
        Constructor method for the Head class
            This defines the layers and structures used in the head

        Create the key, query, and value linear layers
            These are projections that create the Q, K, V matrices
            from the input embeddings
        Creates a buffer for the look-ahead mask
            A buffer is a tensor that stores some of the model's state
            The 'tril' tensor is a lower triangular matrix of ones
            This masks out the future tokens in the sequence
            This is the key difference between decoder and encoder
                self-attention
        Create a dropout layer for regularization

        Args:
            config: GPTConfig
                The hyperparameters for the model
            head_size: int
                The size of the heads (always a whole number)
                This is the embedding size divided by the number of heads
        '''

        super().__init__()

        # Create the key, query, and value linear layers
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)

        # Create the look-ahead mask
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

        # Regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        '''
        The forward pass of the head

        (1) Create the Query, Key, and Value matrices
        (2) Compute attention scores, also known as "affinities"
            Matrix multiplication of Q and K
                The last two dimensions (-2, -1) of K are transposed
                K changes from (B, T, hs) to (B, hs, T)
                The transpose is necessary for the matrix multiplication
            Attention scores are scaled by the square root of the key dimension
                This is needed to prevent exploding or vanishing gradients
        (3) Apply a look-ahead mask to the attention scores
            This is so the transformer cannot 'see' future tokens
            We have self.tril, which is a precomputed lower triangular matrix
            This process slices the matrix to get the right size for the tensor
            This creates a tensor where future positions are masked out
        (4) Apply softmax to the attention scores
            This is to create a probability distribution
            This represents the weight of attention that a token should
                give to other tokens
        (5) Regularization using dropout
        (6) Get the weighted attention
            This is the weighted sum of the values
            Found by matrix multiplication of the probability distribution
                with the value matrix
            Resulting shape is (B, T, hs)

        Args:
            input_tensor: torch.Tensor
            (B, T, C)

        Returns:
            torch.Tensor
            (B, T, head_size)
        '''

        # Create the Query and Key matrices; Shape: (B, T, hs)
        q = self.query(input_tensor)
        k = self.key(input_tensor)
        v = self.value(input_tensor)

        # Compute attention scores
        attention = q @ k.transpose(-2, -1)
        scaled_attention = attention * k.shape[-1] ** 0.5

        # Look-ahead mask (output shape is B, T, T)
        _, T, _ = input_tensor.shape
        masked_attention = scaled_attention.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        # Get the probability distribution (still B, T, T)
        probability_distribution = F.softmax(masked_attention, dim=-1)

        # Regularization
        probability_distribution = self.dropout(probability_distribution)

        # Get the final weighted attention (B, T, hs)
        weighted_attention = probability_distribution @ v

        return weighted_attention


class MultiHeadAttention(nn.Module):
    '''
    The multi-headed self-attention mechanism
    Creates multiple heads of self-attention in parallel

    The multiple heads can learn different relationships between tokens
        It's like they all have a different perspective on the tokens
        They each learn different relationships

    Methods:
        __init__:
            Constructor method for the MultiHeadAttention class
            Create multiple heads and a projection layer
        forward:
            The forward pass of the multi-headed self-attention mechanism
    '''

    def __init__(
        self,
        config: 'GPTConfig',
        head_size: int
    ) -> None:
        '''
        Constructor method for the MultiHeadAttention class

        (1) Create multiple heads (modules) and store them in a list
        (2) Create a projection layer to combine the heads
        (3) A dropout layer for regularization

        Args:
            config: GPTConfig
                The hyperparameters for the model
            head_size: int
                The size of the heads
                This is the embedding size divided by the number of heads
        '''

        super().__init__()

        # Create multiple heads (modules) and store them in a list
        self.heads = nn.ModuleList(
            [
                Head(
                    config,
                    head_size
                )
                for _ in range(config.n_head)
            ]
        )

        # Combines the results of the heads, and projects them back to
        #   the original tensor size
        self.proj = nn.Linear(
            head_size * config.n_head,
            config.n_embd
        )

        # A dropout layer for regularization
        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        input_sequence: torch.Tensor
    ) -> torch.Tensor:
        '''
        The forward pass of the multi-headed self-attention mechanism

        (1) Pass the input sequence through each of the heads
        (2) Concatenate the results of the heads
        (3) Project the concatenated tensor back to the original size

        Args:
            input_sequence: torch.Tensor
                The input sequence (B, T, C)

        Returns:
            torch.Tensor
                The projected output tensor (B, T, C)
        '''

        # Pass the input sequence through each of the heads
        #   Results are concatenated along the last dimension (dim=-1)
        #   The last dimension is the number of heads
        concatenated_heads_output = torch.cat(
            [attention_head(input_sequence) for attention_head in self.heads],
            dim=-1
        )

        # Project the concatenated tensor back to the original size
        #   Then apply dropout for regularization
        projected_output = self.dropout(
            self.proj(concatenated_heads_output)
        )

        return projected_output


class FeedFoward(nn.Module):
    '''
    The feed-forward transformation in the decoder block

    Expands the dimensionality of the embeddings
        This allows for more complex relationships between tokens
    Adds non-linearity to the model
        Using an activation function (ReLU)
    Reduces the dimensionality back to the original size
    '''

    def __init__(
        self,
        config: 'GPTConfig'
    ) -> None:
        '''
        Constructor method for the FeedFoward class

        Create a simple neural network with two linear layers
            The first layer expands the dimensions
            The second layer reduces the dimensions back to the original size
        ReLU activation function is used for non-linearity
        Dropout is used for regularization

        Args:
            config: GPTConfig
                The hyperparameters for the model
        '''

        super().__init__()

        # The feed-forward network
        self.ffn = nn.Sequential(
            # Layer 1: Expand the dimensions 4x
            nn.Linear(config.n_embd, 4 * config.n_embd),

            # ReLU activation function
            nn.ReLU(),

            # Layer 2: Reduce the dimensions back to the original size
            nn.Linear(4 * config.n_embd, config.n_embd),

            # Dropout for regularization
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        '''
        The forward pass of the feed
        Pass the input tensor through the feed-forward network

        Args:
            input_tensor: torch.Tensor
                The input tensor (B, T, C)

        Returns:
            torch.Tensor
                The output tensor (B, T, C)
        '''

        return self.ffn(input_tensor)


class Block(nn.Module):
    '''
    A complete decoder block
    The model will use several of these as layers

    A decoder block consists of:
        Masked multi-headed self-attention
        Layer Normalization
        Point-wise feed-forward transformation
        Residual connections around each of the sub-layers

    Methods:
        __init__:
            Constructor method for the Block class
            Build the architecture of the block using sub-layers
        forward:
            The forward pass of the block
    '''

    def __init__(
        self,
        config: 'GPTConfig'
    ) -> None:
        '''
        Constructor method for the Block class
        Build the architecture of the block using sub-layers

        The Heads build a deep understanding of the embeddings,
            and how they relate to each other
        The FFN is a simple neural network
            It expands the dimensions, adds non-linearity, and reduces
            the dimension back to the original size
            Each embedding is processed independently
        Layer normalization is is a standard PyTorch LayerNorm layer
            It is used for stability, preventing exploding gradients

        Args:
            config: GPTConfig
                The hyperparameters for the model
        '''

        super().__init__()

        # Multi-headed self-attention layer
        self.sa = MultiHeadAttention(
            config=config,
            # Head size is the embedding size divided by the number of heads
            head_size=config.n_embd // config.n_head,
        )

        # The feed-forward layer
        self.ffwd = FeedFoward(config)

        # Layer normalization
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        '''
        The forward pass of the block
        (1) Computes the self-attention
            This applies normalization and residual connections
        (2) Computes the feed-forward transformation
            Also applies normalization and residual connections
        '''

        # Self attention, normalization, and residual connection
        x = x + self.sa(self.ln1(x))

        # Feed-forward transformation, normalization, and residual connection
        x = x + self.ffwd(self.ln2(x))

        return x


class GPTLanguageModel(nn.Module):
    '''
    The GPT model for language modeling
    Builds a transformer using other classes as building blocks
    The GPTConfig class is used to track hyperparameters and GPT architecture

    Methods:
        __init__:
            Initializes the GPT model
        _init_weights:
            Xavier/Glorot initialization of the weights
        forward:
            The forward pass of the model
            Training or inference
        generate:
            Generate new tokens from the model
        save_checkpoint:
            Save the model checkpoint
        load_checkpoint:
            Load a model checkpoint
    '''

    def __init__(
        self,
        config: 'GPTConfig',
        vocab_size: int,
    ) -> None:
        '''
        Initializes the GPT model.

        Sets up the GPT model for language modeling by:
            (1) Create an embedding layer for the tokens
                Tracks tokens semantic meanings
            (2) Create an embedding layer for the positions
                Understand the sequence of tokens
            (3) Create a series of transformer blocks
            (4) Create a final normalization layer
                Stabalize activation functions
                Keeps values in a reasonable range
            (5) Create a linear layer for the output
                Maps the embeddings to the vocabulary size
                Producing logits for the next token prediction.
            (6) Initialize the weights of the model

        Args:
            config: GPTConfig
                The hyperparameters for the model
                'config' helps add flexibility to the model

            vocab_size: int
                The size of the vocabulary
                This is the number of unique tokens in the dataset
        '''

        super().__init__()
        self.device = config.device
        self.block_size = config.block_size
        self.pad_token = config.pad_token

        # Embedding layer for tokens; Essentially a 2D tensor
        #   A trainable matrix where each row represents a token
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            config.n_embd
        )

        # An embedding layer for token positions; 2D tensor
        #   Trainable matrix to understand the sequence of tokens
        self.position_embedding_table = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        # Create a series of transformer blocks (core of the transformer)
        self.blocks = nn.Sequential(
            *[
                Block(config)
                for _ in range(config.n_layer)
            ]
        )

        # Normalization layer
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Linear transformation layer
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        # Initialize the weights of the model
        self.apply(self._init_weights)

        # Get the parameter count of the model
        self.param_count = sum(
            # numel() is a PyTorch way to count elements in a tensor
            p.numel()
            for p in self.parameters()
        )

    def _init_weights(
        self,
        module: nn.Module
    ) -> None:
        '''
        Xavier/Glorot initialization of the weights for a module
            A non-zero value helps training

        If there is no bias values on the linear layer, set to zero
            Embedding layers do not have bias values

        Note, this can't be used on every module
            Some modules are the wrong size, and will retain default
            Pytorch initialization values

        Args:
            module: nn.Module
                The module to initialize the weights for
        '''

        # Check if the module has 'weight' attribute and the weight
        #   has at least 2 dimensions
        if (
            (hasattr(module, 'weight') and module.weight is not None) and
            (module.weight.dim() >= 2)
        ):
            torch.nn.init.xavier_normal_(module.weight)

        # Initialize bias to zero if the module is an instance
        # of nn.Linear and has a bias
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        The forward pass of the model
            When the model is called, the forward pass is executed
            This could be for training or inference
            No targets are passed for inference

        This method works heavily with tensors. Three dimensions are used:
            B: Batch size
            T: Sequence length (AKA 'time')
            C: Number of channels (or dimensions in the embedding)

        Tensors are passed through the layers that were defined in __init__
            Embeddings, blocks, normalization, and linear layers

        Loss is calculated with the cross-entropy loss function
            Requires the logits and targets to be in a particular shape
            This ignores the 'pad' token when learning

        Args:
            idx: torch.Tensor
                A batch of indices of the tokens in the sequence
                Shape: (B, T)

            targets: torch.Tensor
                The indices of the tokens in the target sequence
                Shape: (B, T)
                Not needed for inference

        Returns:
            A tuple of (logits, loss):
            logits: torch.Tensor
                The model's predictions for the next token in each sequence.
                Shape: (B, T, vocab_size).
            loss: torch.Tensor or None
                The calculated loss if targets are provided.
                None during inference
        '''

        # Get the batch size and sequence length as B, T
        B, T = idx.shape

        # 'idx' is a tensor, with a batch of token numbers (B, T)
        #   Each token is mapped to an embedding vector
        #   This creates a new token embedding tensor (B, T, C)
        tok_emb = self.token_embedding_table(idx.long())  # (B,T,C)

        '''
        Creates a 2D tensor (T, C) of position embeddings for the sequence
        torch.arrange creates a sequence of numbers in a tensor
            In this case, from 0 to T-1 (sequence length)
            This is passed to the position embedding table
        Reshaping
            Each row is a vector that represents a position in the sequence
            The number of rows (T) corresponds to the sequence length
            The number of columns (C) corresponds to the embedding size
        '''
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )

        # Combine the token and position embeddings (B, T, C)
        transformed_embeddings = tok_emb + pos_emb

        # Pass the embeddings through the transformer blocks
        transformed_embeddings = self.blocks(transformed_embeddings)

        # Pass through the normalization layer
        transformed_embeddings = self.ln_f(transformed_embeddings)

        # Pass through the linear layer to get the logits
        #   Predicts the next token in the sequence (B, T, vocab_size)
        logits = self.lm_head(transformed_embeddings)

        # Calculate the loss, unless inferencing
        if targets is None:
            loss = None

        # Training, so calculate the loss
        else:
            # Reshape logits and targets tensors to suit the loss function
            #   Input (logits) should be 2D
            #   Target should be 1D
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T).long()

            # Calculate the loss
            #   The ignore_index is a special token that should be ignored
            #   ([Pad] token in our case)
            loss = F.cross_entropy(
                input=logits,
                target=targets,
                ignore_index=self.pad_token
            )

        return logits, loss

    def generate(
        self,
        context: torch.Tensor,
        max_new_tokens: int
    ) -> torch.Tensor:
        '''
        Generate new tokens from the model
        Accepts an initial sequence of tokens and generates new tokens

        Loop through the number of tokens to generate
            Use an existing sequence of tokens to predict the next token
            We have a context window to adhere to; This is the block size
            Select the last entry in the sequence (the prediction)
            Convert logits to probabilities
            Select a token from the list, based on probabilities
                Not always the highest in the probability distribution

        Args:
            idx: torch.Tensor
                The initial sequence of tokens; Shape: (B, T)
                Pass 'None' to generate a sequence from scratch
            max_new_tokens: int
                The number of tokens to generate

        Returns:
            torch.Tensor
                The sequence of tokens
                Shape: (B, T + max_new_tokens)
        '''

        # Create a blank context if none is provided
        if context is None:
            context = torch.zeros(
                (1, 1),
                dtype=torch.long,
                device=self.device
            )

        # Loop through the number of tokens to generate
        for _ in range(max_new_tokens):
            # Just get the last 'block_size' tokens (context window)
            idx_cond = context[:, -self.block_size:]

            # Get predictions with the forward pass (inferencing); B, T, C
            logits, _ = self(idx_cond)

            # Select just the B and C dimensions for the last entry
            #   The last token is the one we want to predict
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Select a token from the list, based on probabilities; Shape: B, 1
            idx_next = torch.multinomial(probs, num_samples=1)

            # Update the sequence of tokens with the new token and repeat
            context = torch.cat((context, idx_next), dim=1)

        # Finally, return the sequence of tokens
        return context

    def save_checkpoint(
        self,
        filename: str = 'model.pth',
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        epoch: int = None,
        loss_history: dict = None
    ) -> bool:
        '''
        Save the model
        This collects:
            The model state dictionary
            The optimizer state dictionary
            The scheduler state dictionary
            The current epoch number

        These extra parameters are needed to resume training

        Args:
            filename: The filename for saving the checkpoint.
            optimizer: The optimizer used during training.
            scheduler: The learning rate scheduler used during training.
            epoch: The current epoch number.
            loss_history: A dictionary of the loss history.

        Returns:
            bool
                True if the model was saved successfully
                False if the model was not saved
        '''

        # Check all our parameters are provided
        if optimizer is None:
            print('Optimizer not provided. Skipping save...')
            return False

        if scheduler is None:
            print('Scheduler not provided. Skipping save...')
            return False

        if epoch is None:
            print('Epoch not provided. Skipping save...')
            return False

        if loss_history is None:
            print('Loss history not provided. Skipping save...')
            return False

        # Collect all the information we need to save
        print(f'Saving at epoch {epoch + 1}')
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'loss_history': loss_history,
        }

        # Save the model
        try:
            torch.save(checkpoint, filename)
        except Exception as e:
            print(f'Error saving model: {e}')
            return False

        return True

    def load_checkpoint(
        self,
        filename: str = 'model.pth',
        optimizer=None,
        scheduler=None
    ) -> Tuple[int, dict] | None:
        '''
        Load a checkpoint
            Restore the model, optimizer, and scheduler states.

        Args:
            filename: The filename of the checkpoint to load.
            model: The model object to load the state into.
            optimizer: The optimizer object to load the state into
            scheduler: The scheduler object to load the state into

        Returns:
            epoch: int
                The epoch number to resume training from.
            loss_history: dict
                The loss history to resume training from.
        '''

        # Load the checkpoint file
        try:
            checkpoint = torch.load(filename)
        except Exception as e:
            print(f'Error loading model: {e}')
            return None

        # Load the model state
        try:
            self.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f'Error loading model state: {e}')
            return None

        # Optionally load the optimizer and scheduler states
        epoch = 0
        try:
            if optimizer and scheduler:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                epoch = checkpoint.get('epoch', None)
                loss_history = checkpoint.get('loss_history', None)
        except Exception as e:
            print(f'Error loading optimizer and scheduler states: {e}')
            return None

        return epoch, loss_history


class GPTConfig():
    '''
    Configuration for GPT model
    Stores model hyperparameters
        This means less passing of arguments to functions
    '''

    def __init__(
        self,
        device: torch.device,
        tokenizer: 'ChessTokenizer',
        batch_size: int = 64,
        block_size: int = 256,
        n_embd: int = 384,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.2,
        pad_token: int = 0,
    ) -> None:
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

        # Hardware
        self.device = device

        # Tokenizer
        self.tokenizer = tokenizer
        self.pad_token = pad_token

        # Architecture
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        # Training
        # self.eval_iters = eval_iters

        # Regularization
        self.dropout = dropout


if __name__ == '__main__':
    print('This is a module with classes for the transformer model')
    print('Please run model.py for the full model')
