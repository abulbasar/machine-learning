from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import logging


logger = logging.getLogger()


"""
$ curl -i \
--header "Accept: application/json" \
--request POST \
--data '{"age": 33, "bmi": 27.0, "children": 0, "smoker": "yes", "gender": "male", "region": "northeast"}' \
http://127.0.0.1:5001/ 

"""

app = Flask(__name__)
api = Api(app)

count = 0

class InsuranceCalculatorRequest(Resource):
    parser = (reqparse
        .RequestParser(bundle_errors=True)
        .add_argument('age', type = int)
        .add_argument('bmi', type = float)
        .add_argument('children', type = int)
        .add_argument('smoker', type = str)
        .add_argument('gender', type = str)
        .add_argument('region', type = str))
        
        
    def get(self):
        global count
        count += 1 
        return count
        
    def post(self):
        print("input")
        args = self.parser.parse_args(strict=True)
        print(args)
        
        return args

api.add_resource(InsuranceCalculatorRequest, '/')

if __name__ == '__main__':
    app.run(debug=True, port = 5001)