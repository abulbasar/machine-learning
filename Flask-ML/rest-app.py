from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

import pandas as pd
import pickle

import os.path
import sys

"""
Start the rest api service
$ python rest-app.py

Send rest api call using curl command
$ curl -i \
--header "Content-type: application/json" \
--request POST \
--data '{"age": 33, "bmi": 27.0, "children": 0, "smoker": "yes", "gender": "male", "region": "northeast"}' \
http://127.0.0.1:5001/ 


Response should contain prediction: 
{
    "age": 33,
    "bmi": 27.0,
    "children": 0,
    "smoker": "yes",
    "gender": "male",
    "region": "northeast",
    "prediction": 21871.30625871111
}

"""

app = Flask(__name__)
api = Api(app)


class InsuranceCalculator(Resource):
    
    def __init__(self):
        
        # Load the model. If the model does not exist, run train.py to build the model.
        model_path = "/tmp/model.pickle"

        if not os.path.isfile(model_path):
            print("""
            [Error] Saved model is not found {}.
            First run train.py to build the model and relaunch flask server.\n\n
            """.format(model_path))
            sys.exit(1)
            
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
    
    # Function to predict single input record
    def predict(self, record):
        df_input = pd.DataFrame.from_dict([record])
        prediction = 10 ** self.model.predict(df_input)
        return prediction[0]
    
    post_parser = (reqparse
        .RequestParser(bundle_errors=True)
        .add_argument('age', type = int)
        .add_argument('bmi', type = float)
        .add_argument('children', type = int)
        .add_argument('smoker', type = str)
        .add_argument('gender', type = str)
        .add_argument('region', type = str))
        
    def post(self):
        args = self.post_parser.parse_args(strict = True)
        print(args)
        args["prediction"] = self.predict(args)
        return args
        

api.add_resource(InsuranceCalculator, '/')

if __name__ == '__main__':
    app.run(debug=True, port = 5001)
    
    
    