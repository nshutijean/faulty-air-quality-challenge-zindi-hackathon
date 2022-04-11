import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# flask app
app = Flask(__name__)

# model = pickle.load(open('model.pkl', 'rb'))

# app.route("/")
# def hello():
#     return "hello"

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        req_json = request.json
        req_data = pd.DataFrame(req_json)
        y_pred = model.predict(req_data)

        if isinstance(y_pred, np.int64):
            y_pred = int(y_pred)
        # elif isinstance(y_pred, np.float64):
        #     y_pred = float(y_pred)

        return jsonify({"data": list(y_pred)})

# if __name__ == "__main__":
#     app.run(debug=True)
