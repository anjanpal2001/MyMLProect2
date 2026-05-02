from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline    


application=Flask(__name__)
# app=application

## Route fro a home page
@application.route('/')
def index():
    return render_template('index.html')
@application.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    application.run(debug=True,host="0.0.0.0",port=8080)  
    
    
    # This is my full deployment code for the flask application. I have created a flask application and defined two routes. The first route is for the home page and the second route is for the prediction page. In the prediction page, I am taking the input from the user and creating a dataframe from it. Then I am using the predict pipeline to get the results and rendering the results on the home page.
    
    