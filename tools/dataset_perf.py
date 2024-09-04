'''
Performance profiling of the dataset class
Sets up a dummy class to profile without all the training overhead
'''

import sys
import os
import math
import cProfile
import pstats

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import ManageDataSet
from transformer_blocks import GPTConfig
from config import Config
from tokenizer import ChessTokenizer


def train(
    dataset: ManageDataSet,
    percent: float = 1.0,
) -> None:
    # Start loading the data in a separate thread
    dataset.start_data_loading(percentage=percent)

    for _ in range(math.ceil(1.0 / percent)):
        # Fetch the next chunk from the queue
        dataset.data_queue.get()

    # End of loop, stop the thread
    dataset.stop_data_loading()


if __name__ == '__main__':
    # Read the configuration file
    settings = Config(config_file='config.yaml')

    # Set up the tokenizer
    tokenizer = ChessTokenizer()
    tokenizer.load()

    # Set up the model config
    model_config = GPTConfig(
        device=None,
        tokenizer=tokenizer,
        batch_size=None,
        block_size=None,
        n_embd=None,
        n_head=None,
        n_layer=None,
        dropout=None,
        pad_token=None,
    )

    chess_dataset = ManageDataSet(
        model_config=model_config,
        dataset_dir=settings.dataset['path'],
    )

    cProfile.run(
        'train(chess_dataset, percent=1.0)',
        './tools/profile_output'
    )

    # train(chess_dataset, percent=0.1)

    # Print the profiling results
    with open('./tools/profile_results.txt', 'w') as f:
        p = pstats.Stats('./tools/profile_output', stream=f)
        p.sort_stats('cumulative').print_stats(10)
