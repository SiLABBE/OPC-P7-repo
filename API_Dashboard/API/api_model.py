# -*- coding: utf-8 -*-

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from customer import customer
import numpy as np
import joblib
import shap

import pandas as pd
# 2. Create the app object
app = FastAPI()
pipeline = joblib.load("pipe_lr_model_selected.joblib")
shap_explainer = joblib.load(filename='shap_explainer_selected.bz2')

# 2. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'API for loan customer prediction'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted score with the confidence
@app.post('/predict')
def predict_Customer(data:customer):
    data = data.dict()
    X_customer=[data['customer_data']]

    pred = pipeline.predict_proba(X_customer)

    X_std = pipeline[0].transform(X_customer)

    shap_values = shap_explainer(X_std[0:1])

    return {pred[0][0]}, {tuple(shap_values.values[0])}, {shap_values.base_values[0]}

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn api_model:app --reload