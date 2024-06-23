import gradio as gr
import tkinter as tk
from tkinter import filedialog
import os
from tokenizer import ChessTokenizer
import random


bye_world = gr.Interface(lambda name: "Bye " + name, "text", "text")


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
    overwrite=False,
):
    # Create the tokenizer
    tokenizer = ChessTokenizer()

    # Select the files to use for training
    full_file_list = os.listdir(path)
    random.shuffle(full_file_list)
    file_list = full_file_list[:int(len(full_file_list) * (percentage / 100))]

    print(
        f"Using {len(file_list)} out of {len(full_file_list)} files\
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
        overwrite=overwrite
    )


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
        chk_token_overwrite = gr.Checkbox(
            label="Overwrite existing files",
            value=False,
            info="If checked, the existing JSON files will be overwritten. \
                If unchecked, the files will be saved with a timestamp."
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
                chk_token_overwrite,
            ]
        )


# The training tab
with gr.Blocks() as train_tab:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=update, inputs=inp, outputs=out)


# The generator tab
with gr.Blocks() as gen_tab:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=update, inputs=inp, outputs=out)


front_end = gr.TabbedInterface(
    [token_tab, train_tab, gen_tab],
    ["Tokenizer", "Trainer", "Generator"]
)

if __name__ == "__main__":
    front_end.launch()
