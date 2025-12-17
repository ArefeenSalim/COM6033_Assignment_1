from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import calendar

app = Flask(__name__)

model = joblib.load("model/uk_house_price_model.joblib") # loads the trained machine learning model

# maps form inputs to the encoded values
def map_inputs(form):
    property_map = {
        "Detached": "D",
        "Semi-Detached": "S",
        "Terraced": "T",
        "Flat": "F",
        "Other": "O"
    }
    new_build_map = {"Yes": "Y", "No": "N"}
    tenure_map = {"Freehold": "F", "Leasehold": "L"}

    return {
        "property_type": property_map[form["property_type"]],
        "new_build": new_build_map[form["new_build"]],
        "tenure": tenure_map[form["tenure"]],
        "county": form["county"].strip(),
        "year": int(form["year"]),
        "month": int(form["month"])
    }


@app.route("/", methods=["GET"]) # defines the home page route
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"]) # defines the prediction route
def predict():
    try:
        month_num = int(request.form["month"]) # converts numeric month input into a readable month name for display
        month_name = calendar.month_name[month_num]
        # stores user-friendly input values to display on the results page
        display_inputs = {
            "Property Type": request.form["property_type"],
            "New Build": request.form["new_build"],
            "Tenure": request.form["tenure"],
            "County": request.form["county"],
            "Year": request.form["year"],
            "Month": month_name
        }
        # maps form values to the encoded format required by the trained model
        property_map = {
            "Detached": "D",
            "Semi-Detached": "S",
            "Terraced": "T",
            "Flat": "F",
            "Other": "O"
        }
        new_build_map = {"Yes": "Y", "No": "N"}
        tenure_map = {"Freehold": "F", "Leasehold": "L"}
        model_inputs = {
            "property_type": property_map[request.form["property_type"]],
            "new_build": new_build_map[request.form["new_build"]],
            "tenure": tenure_map[request.form["tenure"]],
            "county": request.form["county"].strip(),
            "year": int(request.form["year"]),
            "month": int(request.form["month"]),
        }

        X_input = pd.DataFrame([model_inputs]) # converts inputs into a DataFrame so it matches the model input format

        pred_log = model.predict(X_input)[0] # predicts the log-transformed price using the trained model
        pred_price = float(np.expm1(pred_log)) # converts the log price back to the original price
        # renders the results page with the prediction and user inputs
        return render_template(
            "result.html",
            prediction=f"£{pred_price:,.0f}",
            inputs=display_inputs
        )

    except Exception as e:
        # displays an error message
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
