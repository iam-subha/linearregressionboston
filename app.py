import pickle
import json
from flask import Flask, app,jsonify,url_for,render_template,request,redirect
import numpy as np
import pandas as pd
app=Flask(__name__)
regmodel=pickle.load(open("lrgmod.pkl", "rb"))
scalar=pickle.load(open("scale.pkl","rb"))
@app.route('/')
def home():
    return render_template("home.html")
@app.route("/predict_api",methods=['POST'])
def predict_api():
    data=request.json["data"]
    print (data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform((np.array(list(data.values())).reshape(1,-1)))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])
@app.route("/predict",methods=['POST'])
def predict():
    data=[float(i) for i in request.form.values()]
    fin_input=scalar.transform((np.array(data).reshape(1,-1)))
    print(fin_input)
    fin_output=regmodel.predict(fin_input)[0]
    return render_template("home.html",prediction_text=f"The predicted price is {fin_output} ")
    
if __name__ == '__main__':
    app.run(debug=True)
    


