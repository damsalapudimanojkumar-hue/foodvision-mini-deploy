"""
ui.py — the Gradio interface definition, kept separate from model/inference
code so it can be taught on its own: this is how any predict() function plugs
into a Gradio UI, independent of what the model actually does.
"""
import gradio as gr


def build_demo(predict, example_list):
    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes=3, label="Prediction"),
            gr.Number(label="Prediction Time (s)"),
        ],
        examples=example_list,
        title="Pizza, Steak, or Sushi?",
        description="Upload a photo and the model will classify it as pizza, steak, or sushi.",
    )
