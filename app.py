from flask import Flask, jsonify, redirect, request, render_template
import pickle
import numpy as np
import yaml
import json
with open("config.yaml", "r") as stream:
    file_paths = yaml.safe_load(stream)
app = Flask(__name__)


def get_list_of_inputs(method):
    if method == "POST":
        data = request.form
    else:
        data = request.args
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
        list_of_inputs = get_list_of_inputs(method="POST")
        input_array = np.array(list_of_inputs, dtype=np.float32).reshape(1, 7)
        model = pickle.load(open(file_paths['model_path'], 'rb'))
        prediction = model.predict(input_array)
        print(prediction)
        return render_template('index.html', title='Home', prediction=prediction.tolist())
    else:
        return render_template('index.html', title='Home')


@app.route('/getpredictions', methods=['GET'])
def index_json():
    if request.method == "GET":
        list_of_inputs = get_list_of_inputs(method="GET")
        input_array = np.array(list_of_inputs, dtype=np.float32).reshape(1, 7)
        model = pickle.load(open(file_paths['model_path'], 'rb'))
        prediction = model.predict(input_array)
        return jsonify(prediction=prediction.tolist())


@app.route('/dashboard')
def dashboard():
    return redirect('http://192.168.1.66:8501/')


if __name__ == "__main__":
    app.run(debug=True)
