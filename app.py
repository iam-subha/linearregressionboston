import pickle
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
if __name__ == '__main__':
    app.run(debug=True)
    


