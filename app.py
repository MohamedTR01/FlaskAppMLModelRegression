import pandas as pd
import numpy as np
import sklearn
import joblib
from sklearn.linear_model import LinearRegression
from flask import Flask,render_template,request




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method =='POST':
        print(request.form.get('Années'))
        print(request.form.get('Kilométrage'))
        print(request.form.get('Le nombre de chevaux'))

        try:
            var_1=float(request.form['Années'])
            var_2=float(request.form['Kilométrage'])
            var_3=float(request.form['Le nombre de chevaux'])
            pred_args=[var_1,var_2,var_3]
            pred_arr=np.array(pred_args)
            preds=pred_arr.reshape(1,-1)
            model=open("linear_regression_model.pkl","rb")
            lr_model=joblib.load(model)
            model_prediction=lr_model.predict(preds)
            model_prediction=round(float(model_prediction),2)
        except ValueError:
            return render_template('Erreur.html')
    return render_template('predict.html',prediction=model_prediction)






if __name__ == '__main__':
    app.run(debug=True)