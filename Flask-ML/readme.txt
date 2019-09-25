Install the required libraries. Skip this step, if not neccessary. 
$ pip install -r -U requirements.txt

$ conda update --all


Train the model for insurance price prediction and save the model. Change the path where you want to save the model. Use the same path in app.py and rest-app.py files as well.
$ python train.py

The saved model is loaded from the disk. The reloaded model can be used without further training. Training model is expensive process (it takes time and $$$)

Start the flask application server
$ python app.py


After starting the app server, open this link in the browser http://localhost:5000
Enter a sample record and submit to receive an estimated price.


Train and save the model. Skip this step, if you have already trained the model.
$ python train.py

Start the rest api service. Keep the service running.
$ python rest-app.py

Open another terminal and send rest api call using curl command
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

