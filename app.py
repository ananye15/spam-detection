import numpy as np
from flask import Flask, render_template,request
import pickle
import pandas as pd

#Initialize the flask App


app = Flask(__name__)

model = pickle.load(open('spam_detection.pkl', 'rb'))
extracted_feature=pickle.load(open('tf-transform.pkl','rb'))
@app.route('/')
def home():
    return render_template('test.html')


@app.route('/predict',methods=['POST'])
def predict():
   
    #For rendering results on HTML GUI
    if request.method ==  'POST':
        message=request.form['message']
        data=[message]
        vect=extracted_feature.transform(data).toarray()
        my_prediction=model.predict(vect)
        return render_template('result.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run(debug=True)		
