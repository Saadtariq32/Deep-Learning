#ChatGPT used to adapt previous example.
import tkinter as tk
import pandas as pd
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model("ANN Classification/ann_classification_model.keras")

# Feature configuration
numeric_features = [
    'credit_score', 'age', 'tenure', 'balance',
    'products_number', 'credit_card', 'active_member', 'estimated_salary'
]
gender_feature = 'gender'
country_features = ['country_Germany', 'country_Spain']

# Output class labels (0 -> No churn, 1 -> Churn)
labels = ["No", "Yes"]

#Create GUI window
window = tk.Tk()
window.title("Customer Churn Prediction (ANN Model)")
window.option_add("*font", "lucida 20 bold")

entries = {}

# Numeric feature entries
for feat in numeric_features:
    label = tk.Label(window, text=feat.replace('_', ' ').title())
    label.pack()
    entry = tk.Entry(window)
    entry.pack(pady=5)
    entries[feat] = entry

# Gender radio buttons
gender_var = tk.IntVar(value=0)
tk.Label(window, text="Gender").pack()
tk.Radiobutton(window, text="Female", variable=gender_var, value=0).pack()
tk.Radiobutton(window, text="Male", variable=gender_var, value=1).pack()

# Country toggles
country_vars = {}
for feat in country_features:
    var = tk.IntVar(value=0)
    tk.Label(window, text=feat.replace('_', ' ').title()).pack()
    tk.Radiobutton(window, text="No", variable=var, value=0).pack()
    tk.Radiobutton(window, text="Yes", variable=var, value=1).pack()
    country_vars[feat] = var

# Prediction result label
result_var = tk.StringVar()
tk.Label(window, textvariable=result_var).pack(pady=20)

# Prediction function
def set_text_by_button():
    input_data = {}
    try:
        # Collect numeric inputs
        for feat in numeric_features:
            input_data[feat] = float(entries[feat].get())
        # Gender
        input_data[gender_feature] = gender_var.get()
        # Country
        for feat in country_features:
            input_data[feat] = country_vars[feat].get()
    except Exception as e:
        result_var.set(f"Error: {e}")
        return

    # Match feature order from training
    input_order = [
        'credit_score', 'gender', 'age', 'tenure', 'balance',
        'products_number', 'credit_card', 'active_member',
        'estimated_salary', 'country_Germany', 'country_Spain'
    ]

    sample = pd.DataFrame([input_data])[input_order]

    # Make prediction
    prediction_probs = model.predict(sample)[0]
    prediction = int(prediction_probs.argmax())

    # Display result
    result_var.set(
        f"Did this person churn? {labels[prediction]}\n"
        f"Probability No: {prediction_probs[0]:.2f}, Yes: {prediction_probs[1]:.2f}"
    )

# Predict button
tk.Button(window, height=1, width=20, text="Predict Churn", command=set_text_by_button).pack(pady=20)

# Run GUI loop
window.mainloop()
