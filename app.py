import yaml
import pickle
import numpy as np
from flask import jsonify
from flask import Flask, redirect, request, render_template
app = Flask(__name__)

with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)


def load_the_model(model_path):
    return pickle.load(open(model_path, 'rb'))


def get_prediction(input_list):
    input_array = np.array(input_list, dtype=np.float32).reshape(1, 7)
    model = load_the_model(file_paths['model_path'])
    prediction = (model.predict(input_array)).tolist()
    return jsonify(prediction=prediction)


def get_inputs():
    data = request.form
    nitrogen_content = data["n"]
    phosphorus_content = data["p"]
    potassium_content = data["k"]
    ph_value = data["ph"]
    temperature_value = data["temp"]
    humidity_value = data["humid"]
    rainfall_value = data["rain"]
    list_of_inputs = [nitrogen_content, phosphorus_content, potassium_content,
                      ph_value, temperature_value, humidity_value, rainfall_value]
    return list_of_inputs


# http://127.0.0.1:5000/getpredictions?n=90&p=30&k=50&ph=20&temp=40&humid=50&rain=100
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        list_of_inputs = get_inputs()
        response = get_prediction(list_of_inputs)
        return render_template('index.html', title='Home', prediction=response.json['prediction'])
    else:
        return render_template('index.html', title='Home')


@app.route('/dashboard')
def dashboard():
    return redirect('http://192.168.1.66:8501/')


if __name__ == "__main__":
    app.run(debug=True)
