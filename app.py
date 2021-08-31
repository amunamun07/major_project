import yaml
import pickle
import requests
import numpy as np
from flask import jsonify
from flask import Flask, redirect, request, render_template

app = Flask(__name__)

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)


def load_the_model(model_path):
    """Loads and returns the model from the project directory.

    Args:
        model_path: Path of the model

    Returns:
        model. A linear_model loaded from the directory.

    """
    try:
        return pickle.load(open(model_path, "rb"))
    except pickle.UnpicklingError as e:
        raise f"Unpickling Error: {e}"
    except Exception as e:
        raise e


def get_inputs() -> dict:
    """Binds the input queries from the user  to a list followed by dictionary

    Returns:
        dict. A dictionary containing one key with list of inputs

    """
    data = request.form
    nitrogen_content = data["n"]
    phosphorus_content = data["p"]
    potassium_content = data["k"]
    ph_value = data["ph"]
    temperature_value = data["temp"]
    humidity_value = data["humid"]
    rainfall_value = data["rain"]
    list_of_inputs = [
        nitrogen_content,
        phosphorus_content,
        potassium_content,
        ph_value,
        temperature_value,
        humidity_value,
        rainfall_value,
    ]
    return {"list_of_inputs": list_of_inputs}


@app.route("/predict", methods=["POST"])
def get_prediction():
    """predicts the output of the model and returns a json object"""
    list_of_inputs = request.json["list_of_inputs"]
    input_nparray = np.array(list_of_inputs, dtype=np.float32).reshape(1, 7)
    model = load_the_model(file_paths["model_path"])
    prediction = (model.predict(input_nparray)).tolist()
    return jsonify(prediction=prediction)


@app.route("/", methods=["GET", "POST"])
def index():
    """Renders the index page with or without predicted output"""
    if request.method == "POST":
        dict_of_input = get_inputs()
        url = file_paths["flask_url"] + "/predict"
        response = requests.post(url, json=dict_of_input)
        return render_template("index.html", prediction=response.json()["prediction"])
    else:
        return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Redirects to the dashboard"""
    return redirect(file_paths["streamlit_url"])


if __name__ == "__main__":
    app.run(debug=True)
