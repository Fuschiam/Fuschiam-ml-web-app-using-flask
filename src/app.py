from flask import Flask, request, render_template
from pickle import load
import pandas as pd

app = Flask(__name__)
model = load(open("best_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form values (which represent the features)
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])
        val5 = float(request.form["val5"])
        val6 = float(request.form["val6"])
        val7 = float(request.form["val7"])
        val8 = float(request.form["val8"])
        val9 = float(request.form["val9"])
        val10 = float(request.form["val10"])
        val11 = float(request.form["val11"])

        # New data point for prediction (example with some missing values)
        new_data_point_dict = {
            'age': [val1],
            'bmi': [val2],
            'children': [val3],
            'sex_female': [val4],
            'sex_male': [val5],
            'smoker_no': [val6],
            'smoker_yes': [val7],
            'region_northeast': [val8],
            'region_northwest': [val9],
            'region_southeast': [val10],
            'region_southwest': [val11],
        }

        # Convert the dictionary to a DataFrame
        new_data_point_df = pd.DataFrame(new_data_point_dict)

        # Model makes a prediction
        pred_class = model.predict(new_data_point_df)[0]
    else:
        pred_class = None

    return render_template("index.html", prediction=pred_class)

