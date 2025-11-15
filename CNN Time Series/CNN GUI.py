#GUI created with the help of ChatGPT
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import PySimpleGUI as sg
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# LOAD MODEL

model = tf.keras.models.load_model("CNN Time Series/best_model1.keras")
class_names = ["Close price", "Volume"]

# SIGNAL GENERATORS

def generate_close_signal():
    x = np.arange(50)
    y = 0.5 + 0.1*np.sin(x/5) + np.random.normal(0, 0.02, size=50)
    return np.clip(y, 0, 1)

def generate_volume_signal():
    x = np.arange(50)
    y = 0.2 + np.random.normal(0, 0.08, size=50)
    y[10:15] += 0.3
    y[30:35] += 0.3
    return np.clip(y, 0, 1)

# MATPLOTLIB CANVAS HELPERS

def draw_figure(canvas, fig):
    figure_canvas_agg = FigureCanvasTkAgg(fig, canvas.TKCanvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def plot_signal(signal, fig_canvas_elem, fig_agg):
    if fig_agg:
        fig_agg.get_tk_widget().forget()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(signal)
    ax.set_title("Generated Signal")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")

    fig_agg = draw_figure(fig_canvas_elem, fig)
    return fig_agg


# CLASSIFICATION

def classify(signal):
    x = np.expand_dims(signal, axis=(0, 2))  # shape (1, 50, 1)
    pred = model.predict(x)[0]
    class_id = np.argmax(pred)
    confidence = pred[class_id]
    return class_id, confidence


# PySimpleGUI LAYOUT

layout = [
    [sg.Text("CNN Time-Series Classifier", font=("Arial", 18))],

    [sg.Text("Choose signal type:")],
    [sg.Combo(["Close", "Volume"], default_value="Close", key="category", size=(20, 1))],

    [sg.Button("Generate & Classify", size=(20, 1))],
    [sg.Text("", key="result", font=("Arial", 14))],

    [sg.Canvas(key="canvas")]
]

window = sg.Window("Time-Series Classification", layout, finalize=True)

fig_agg = None  # store matplotlib canvas instance


# MAIN LOOP

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break

    if event == "Generate & Classify":
        category = values["category"]

        # Pick generator
        if category == "Close":
            signal = generate_close_signal()
        else:
            signal = generate_volume_signal()

        # Classify
        cls, conf = classify(signal)
        window["result"].update(f"Prediction: {class_names[cls]}  ({conf*100:.2f}% confidence)")

        # Plot
        fig_agg = plot_signal(signal, window["canvas"], fig_agg)


window.close()
