"""
modal_app.py — deploys the existing Gradio app on Modal.

Per Gradio's official Modal deployment guide, the existing app code (model.py,
ui.py, and the setup/predict logic from app.py) needs no modification — this
file only adds the Modal-specific wrapping around it. app.py is untouched and
still works for local testing exactly as before.
"""
import modal

app = modal.App("foodvision-mini")

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "gradio", "fastapi[standard]", "pillow")
    .add_local_python_source("model", "ui")
    .add_local_file(
        "effnetb2_pizza_steak_sushi_20_percent.pth",
        remote_path="/root/effnetb2_pizza_steak_sushi_20_percent.pth",
    )
    .add_local_dir("examples", remote_path="/root/examples")
)

with web_image.imports():
    from pathlib import Path
    from timeit import default_timer as timer

    import torch
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    from model import create_effnetb2_model
    from ui import build_demo


@app.function(image=web_image, max_containers=1)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    device = "cpu"
    class_names = ["pizza", "steak", "sushi"]

    model, auto_transforms = create_effnetb2_model(num_classes=3)
    checkpoint_path = Path("/root/effnetb2_pizza_steak_sushi_20_percent.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    def predict(img):
        start_time = timer()

        img_transformed = auto_transforms(img).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            logits = model(img_transformed)
            probs = torch.softmax(logits, dim=1)

        pred_labels_and_probs = {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}

        pred_time = round(timer() - start_time, 4)
        return pred_labels_and_probs, pred_time

    example_dir = Path("/root/examples")
    example_list = [[str(p)] for p in sorted(example_dir.glob("*.jpg"))] if example_dir.exists() else []

    demo = build_demo(predict, example_list)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
