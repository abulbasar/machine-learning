#!/usr/bin/env python
# coding: utf-8

"""
First run train.py. That scripts train a model for insurance price prediction and save the model.
$ python train.py

The saved model is loaded from the disk. The reloaded model can be used without further training. 

Start the flask application server
$ python app.py


After starting the app server, open this link in the browser http://localhost:5000
Enter a sample record and submit to receive an estimated price.

"""


from flask import Flask, render_template, request, redirect
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
with open("/tmp/model.pickle", "rb") as f:
    model = pickle.load(f)

# Function to predict single input record
def predict(record):
    df_input = pd.DataFrame.from_dict([record])
    prediction = 10 ** model.predict(df_input)
    return prediction[0]


@app.route('/')
def home():
    return render_template('input.html')
    

@app.route('/predict_price',methods = ["GET", "POST"])
def result():
    request_data = None
    prediction = None
    if request.method == "POST":
        request_data = dict(request.form)
        print(request_data)
        prediction = predict(request_data)
        return render_template("output.html", record = request_data, prediction = prediction)
    return redirect("/", code=302)
    
if __name__ == '__main__':
    app.run(debug=True)
    
    
    