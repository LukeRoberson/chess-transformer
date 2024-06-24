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
    Presents a dialog box to select the dataset folder

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

    # Get a list of all training files
    full_file_list = os.listdir(path)
    full_size = len(full_file_list)

    # Check if we can resume training
    if resume and os.path.exists(os.path.join(save_path, "resume.txt")):
        tokenizer.load()

        # Read the resume file
        with open(os.path.join(save_path, "resume.txt"), "r") as f:
            resume = [line.strip() for line in f]

        # Remove files from full_file_list if they exist in 'resume'
        print(f"full:{full_file_list}")
        print(f"resume: {resume}")
        full_file_list = [
            file
            for file in full_file_list
            if file not in resume
        ]

    else:
        resume = False

    # Select the files to use for training
    random.shuffle(full_file_list)
    file_list = full_file_list[:int(len(full_file_list) * (percentage / 100))]

    print(
        f"Using {len(file_list)} out of {full_size} files\
        for tokenizer training."
    )

    # Create the file list
    file_list = [
        os.path.join(path, file)
        for file in file_list
        if os.path.isfile(os.path.join(path, file))
    ]

    # Train the tokenizer
    tokenizer.train(
        file_list=file_list,
        save_path=save_path,
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
        txt_token_dataset = gr.Textbox(
            label="Path to the dataset",
            value=f"{os.getcwd()}\\dataset",
            info="Select the path to the dataset folder"
        )
        btn_token_dataset = gr.Button(
            value="Select Dataset",
        )
        sld_token_dataset = gr.Slider(
            label="Number of files",
            minimum=1,
            maximum=100,
            step=1,
            value=100,
            info="Percentage of files to use for training \
                Files are chosen randomly from the selected directory."
        )
        btn_token_dataset.click(
            fn=dir_selector,
            inputs=txt_token_dataset,
            outputs=txt_token_dataset,
        )

    # Training Group
    with gr.Group():
        txt_token_save = gr.Textbox(
            label="Save location",
            value=f"{os.getcwd()}",
            info="Select the path to save the tokenizer. \
                This is where two files, word2idx.json and idx2word.json, \
                will be saved."
        )
        btn_token_save = gr.Button(
            value="Select save location",
        )
        chk_token_resume = gr.Checkbox(
            label="Resume training",
            value=False,
            info="If checked, training will resume from the selected \
                save location, if a resume file is found. \
                If unchecked, or if there is no resume file, \
                the tokenizer will be trained from scratch."
        )
        btn_token_save.click(
            fn=dir_selector,
            inputs=txt_token_save,
            outputs=txt_token_save,
        )

    # Start Training
    with gr.Group():
        btn_token_start = gr.Button(
            value="Start Training"
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
    ["Tokenizer", "Trainer", "Generator"]
)


# Launch the interface
if __name__ == "__main__":
    front_end.launch()
