---
title: FoodVision Mini
emoji: 🍕
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: false
---

# FoodVision Mini 🍕🥩🍣

Upload a photo and this model classifies it as **pizza**, **steak**, or **sushi**.

It uses an EfficientNet-B2 feature extractor, trained on a 20% split of the
Food101 pizza / steak / sushi images, served through a Gradio interface and
running on CPU.

## Project Structure

```
foodvision-mini-deploy/
├── app.py                                          # loads the model, defines the prediction function, launches the app
├── ui.py                                            # builds the Gradio interface
├── model.py                                         # rebuilds the EfficientNet-B2 architecture
├── effnetb2_pizza_steak_sushi_20_percent.pth        # trained model weights
├── requirements.txt                                 # Python dependencies
├── README.md                                        # this file
└── examples/                                        # sample images to try
    ├── example_pizza.jpg
    ├── example_steak.jpg
    └── example_sushi.jpg
```
