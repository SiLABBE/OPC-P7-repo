# -*- coding: utf-8 -*-

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from customer import customer
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("new_model.pkl","rb")
model = pickle.load(pickle_in)

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

    pred = model.predict_proba(X_customer)

    return {
        pred[0][0]
    }

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn api_model:app --reload