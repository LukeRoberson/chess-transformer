'''
The Gradio front end for the Chess Transformer project
Functions can be run manually, but this makes it easier to use

Run the script to launch the interface,
    then open a browser and go to the URL provided

Contains three tabs:
    (1) Tokenizer: Used to train the tokenizer
    (2) Trainer: Used to train the model
    (3) Generator: Used to generate chess moves
'''

import gradio as gr
import tkinter as tk
from tkinter import filedialog
import os
from tokenizer import ChessTokenizer
import random


# A stub function for tabs that aren't in use yet
def update(name):
    return f"Welcome to Gradio, {name}!"


def dir_selector(dir):
    '''
    Presents a dialog box to select a directory

    Parameters:
        dir (str): The default directory to open the dialog box

    Returns:
        str: The path to the selected folder
    '''

    # If no directory is provided, use the current working directory
    if dir is None or dir == "":
        dir = f"{os.getcwd()}\\dataset"

    # Use tkinter to open a dialog box
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(initialdir=dir)

    return folder_selected


def file_selector(dir):
    '''
    Presents a dialog box to select a file

    Parameters:
        dir (str): The default directory to open the dialog box

    Returns:
        str: The path to the selected file
    '''

    # If no directory is provided, use the current working directory
    if dir is None or dir == "":
        dir = f"{os.getcwd()}\\dataset"

    # Use tkinter to open a dialog box
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askopenfile(initialdir=dir)

    return folder_selected


def train_tokenizer(
    path,
    save_path,
    percentage=100,
    resume=False
):
    '''
    Trains the tokenizer on the dataset
    Requires access to the ChessTokenizer class, which does the real work
    Optionally, we can train on a random subset of the dataset

    Parameters:
        path (str): The path to the dataset folder
        save_path (str): The path to save the tokenizer
        percentage (int): The percentage of files to use for training
        resume (bool): If True, training will resume from the save path
    '''

    # Create the tokenizer
    tokenizer = ChessTokenizer()

    # Train the tokenizer
    tokenizer.train(
        dataset_path=path,
        resume_file=save_path,
        resume=resume,
        percent=percentage,
    )


'''
From this point on, we define the interface
Gradio blocks define the tabs
Within each block, we define groups for each section
'''

# The tokenizer tab
with gr.Blocks() as token_tab:
    # Dataset group
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=3):
                txt_token_dataset = gr.Textbox(
                    label="Path to the dataset",
                    value=f"{os.getcwd()}\\dataset",
                    info="Select the path to the dataset folder"
                )

            with gr.Column(variant='compact', scale=1):
                btn_token_dataset = gr.Button(
                    value="Select Dataset",
                    scale=1,
                )
                btn_token_dataset.click(
                    fn=dir_selector,
                    inputs=txt_token_dataset,
                    outputs=txt_token_dataset,
                )

        with gr.Row():
            sld_token_dataset = gr.Slider(
                label="Number of files",
                minimum=1,
                maximum=100,
                step=1,
                value=100,
                info="Percentage of files to use for training \
                    Files are chosen randomly from the selected directory."
            )

    # Training Group
    with gr.Group():
        # Save location
        with gr.Row():
            with gr.Column(scale=3):
                txt_token_save = gr.Textbox(
                    label="File location",
                    value=f"{os.getcwd()}",
                    info="Select the path for the tokenizer files. \
                        This is where two files, word2idx.json and \
                        idx2word.json, will be saved, or loaded from \
                        if resuming training."
                )

            with gr.Column(scale=1, variant='panel'):
                btn_token_save = gr.Button(
                    value="Select save location",
                    scale=1,

                )
                btn_token_save.click(
                    fn=dir_selector,
                    inputs=txt_token_save,
                    outputs=txt_token_save,
                )

    # Resume options
    with gr.Group():
        with gr.Row():
            chk_token_resume = gr.Checkbox(
                label="Resume training",
                value=False,
                info="If checked, training will resume from the selected \
                    save location, if a resume file is found. \
                    If unchecked, or if there is no resume file, \
                    the tokenizer will be trained from scratch."
            )

        # Resume location
        with gr.Row():
            with gr.Column(scale=3):
                txt_resume = gr.Textbox(
                    label="Resume file",
                    info="Select the path to the resume file, typically \
                    resume.txt. This is created during training, so training \
                    can resume later. If the 'Resume Training' box is \
                    checked, this file will be used to determine which files \
                    have already been trained on.",
                    value=f"{os.getcwd()}\\resume.txt",
                )

            with gr.Column(scale=1, variant='compact'):
                btn_resume = gr.Button(
                    value="Select resume file",
                    scale=1,
                )
                btn_resume.click(
                    fn=dir_selector,
                    inputs=txt_resume,
                    outputs=txt_resume,
                )

    # Start Training
    with gr.Group():
        btn_token_start = gr.Button(
            value="Start Training",
            variant='primary',
        )
        txt_token_train = gr.Textbox(
            label="Training Progress",
        )
        btn_token_start.click(
            fn=train_tokenizer,
            inputs=[
                txt_token_dataset,
                txt_token_save,
                sld_token_dataset,
                chk_token_resume,
            ]
        )


# The training tab - Stub area for now
with gr.Blocks() as train_tab:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=update, inputs=inp, outputs=out)


# The generator tab - Stub area for now
with gr.Blocks() as gen_tab:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=update, inputs=inp, outputs=out)


# The main interface, with the tabs
front_end = gr.TabbedInterface(
    [token_tab, train_tab, gen_tab],
    ["Tokenizer", "Trainer", "Generator"],
    title="Transformer Interface",
)


# Launch the interface
if __name__ == "__main__":
    front_end.launch()
