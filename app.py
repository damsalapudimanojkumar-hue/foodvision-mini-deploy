"""
app.py — the FoodVision Mini web app.

Loads the trained EfficientNet-B2 model and serves it through a Gradio
interface. Everything runs on CPU, matching how the model was timed and
chosen during model-building. Hugging Face Spaces runs this file to keep
the app permanently live.
"""
from pathlib import Path
from timeit import default_timer as timer

import torch

from model import create_effnetb2_model
from ui import build_demo

# --- Setup (runs once when the app starts, not on every request) ---
device = "cpu"
class_names = ["pizza", "steak", "sushi"]

# Rebuild the architecture, then load the trained weights into it
# (the same rebuild-then-load pattern used when the model was saved).
model, auto_transforms = create_effnetb2_model(num_classes=3)

checkpoint_path = Path(__file__).parent / "effnetb2_pizza_steak_sushi_20_percent.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()


# --- Prediction function ---
# Same mechanics used to time the model earlier: preprocess -> eval() ->
# disable gradients with no_grad() -> forward pass -> softmax into probabilities.
def predict(img):
    start_time = timer()

    img_transformed = auto_transforms(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_transformed)
        probs = torch.softmax(logits, dim=1)

    # Gradio's Label component expects a {class_name: probability} dictionary
    pred_labels_and_probs = {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}

    pred_time = round(timer() - start_time, 4)
    return pred_labels_and_probs, pred_time


# --- Example images (loaded from the examples/ folder if it exists) ---
example_dir = Path(__file__).parent / "examples"
example_list = [[str(p)] for p in sorted(example_dir.glob("*.jpg"))] if example_dir.exists() else []


# --- Gradio interface (built in ui.py, kept separate from model/inference code) ---
demo = build_demo(predict, example_list)

# No share=True here: Hugging Face Spaces keeps the app permanently live,
# so Gradio's temporary share link is not needed.
demo.launch()
