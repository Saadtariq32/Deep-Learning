#ChatGPT was used to adapt previous linear regression gui example to adapt to this multi-outpu model as I used a different approach for multi-output this time.
import tkinter
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# loading the model
model = load_model("ANN Regresion/ann_regression_model.keras")

# test row for sample data 
tester_row = {
    'relative_compactness': 0.8,
    'surface_area': 600,
    'wall_area': 270,
    'roof_area': 220,
    'overall_height': 7.0,
    'orientation': 2,
    'glazing_area': 0.1,
    'glazing_area_distribution': 3
}
tester_row = pd.DataFrame([tester_row], dtype=np.float32)

# initial test prediction
result = model.predict(tester_row, verbose=0)
heating_load = float(result[0][0][0])
cooling_load = float(result[1][0][0])
print(f"Predicted Heating Load: {heating_load:.2f}")
print(f"Predicted Cooling Load: {cooling_load:.2f}")

# setup
window = tkinter.Tk()
window.title("Heating & Cooling Load Prediction")
window.option_add("*font", "lucida 20 bold")

entries = {}
features = [
    'relative_compactness', 'surface_area', 'wall_area', 'roof_area',
    'overall_height', 'orientation', 'glazing_area', 'glazing_area_distribution'
]

# creating labels
for feat in features:
    label = tkinter.Label(window, text=feat.replace('_', ' ').title())
    label.pack()
    entry = tkinter.Entry(window)
    entry.pack(pady=5)
    entries[feat] = entry

# labels for predictions
result_var = tkinter.StringVar()
label = tkinter.Label(window, textvariable=result_var)
label.pack(pady=20)


# button to update the predictions
def set_text_by_button():
    input_data = {}
    try:
        # read and convert user inputs
        for feat in features:
            val = entries[feat].get().strip()
            if not val:
                raise ValueError(f"Missing value for {feat.replace('_', ' ')}")

            # handle integers and floats properly
            if feat in ['orientation', 'glazing_area_distribution']:
                input_data[feat] = int(float(val))
            else:
                input_data[feat] = float(val)

        # create DataFrame
        sample = pd.DataFrame([input_data], dtype=np.float32)

        # predict using ANN (two-output model)
        prediction = model.predict(sample, verbose=0)
        heating_load = float(prediction[0][0][0])
        cooling_load = float(prediction[1][0][0])

        # display results in GUI
        result_var.set(
            f"Predicted Heating Load: {heating_load:.2f}\n"
            f"Predicted Cooling Load: {cooling_load:.2f}"
        )

        # print also to console for debugging
        print(f"Predicted Heating Load: {heating_load:.2f}")
        print(f"Predicted Cooling Load: {cooling_load:.2f}")

    except Exception as e:
        result_var.set(f"Error: {e}")
        print(f"Error: {e}")


set_up_button = tkinter.Button(window, height=1, width=20, text="Predict Loads", command=set_text_by_button)
set_up_button.pack(pady=20)

window.mainloop()
