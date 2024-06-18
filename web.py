import gradio as gr


bye_world = gr.Interface(lambda name: "Bye " + name, "text", "text")


def update(name):
    return f"Welcome to Gradio, {name}!"


# The tokenizer tab
with gr.Blocks() as token_tab:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=update, inputs=inp, outputs=out)


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
