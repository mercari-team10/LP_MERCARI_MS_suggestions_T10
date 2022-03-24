import json
import joblib
import pickle
import pandas as pd
import config
import sys
import os
import numpy as np
from dotenv import load_dotenv
import requests
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives.asymmetric import  rsa
import psycopg2
import jwt
from flask import Flask, request

symp_svty = './Symptom-severity.csv'
path_to_model = './gbm_model.pkl'

# preprocessing strps


app = Flask(__name__)


key = None


app = Flask(__name__)

model =  joblib.load(path_to_model)
df_severity = pd.read_csv(symp_svty)
df_severity['Symptom'] = df_severity['Symptom'].str.replace('_',' ')
symptoms = df_severity['Symptom'].unique()


mapper = dict(zip(df_severity.Symptom, df_severity.weight))
 
def encode_data(symp_list) :
    feats = []
    for symp in symp_list :
        feats.append(mapper[symp])
    appends = [0] * (17 - len(feats))
    final_feats = feats  + appends
    final_feats = np.array([final_feats])
    return final_feats
    
def predict_label(x,model) :
    label = model.predict(x)
    return label

def predict_disease(symptoms) :
    feats = encode_data(symptoms)
    disease = predict_label(feats,model)
    return disease[0]

@app.route("/filter_hospitals",methods=['GET'])
def filter_and_sort_hospitals() :
    args = request.get_json()
    symptoms = list(args['symptoms'].split(','))
    pred_disease = predict_disease(symptoms)
    return pred_disease


if __name__ == '__main__':
    app.run()