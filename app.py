from ast import arg
import xgboost as xgb
import streamlit as st
import pandas as pd
import numpy as np
from utils import Preprocessing




# Load the model
model = xgb.XGBClassifier()
model.load_model('model.json')

st.title('Could we survive from Titanic :ship:?')

option = st.selectbox(
    'How you input?',
    ('Input manually', 'Input file')
)

columns = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
result = 0

def predict_manual():
    row = np.array([pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked])
    X = pd.DataFrame([row], columns=columns)
    X = Preprocessing(X)
    return model.predict(X)[0]

    

def predict_file():
    global result
    test_df = Preprocessing(test_feature)
    prediction = model.predict(test_df)
    result_df = pd.DataFrame({'PassengerId': test_index, 'Survived': prediction})
    result_df["Survived"] = np.where(result_df["Survived"] == 1, "Alive", "Dead")
    return result_df
    
if option == 'Input manually': 
    click=False
    name = st.text_input("What's your name?", placeholder='Type Here ...')
    id = st.text_input("What's your passenger ID?", placeholder='Type Here ...')
    sex = st.radio('Choose your gender:', options=["Male", "Female"])
    pclass = st.select_slider('Choose your passenger class:', options=[1, 2, 3])
    age = st.number_input('How old are you?', min_value=0., max_value=100., value=0.)
    sibsp = st.number_input('How many siblings or spouses do you have?', min_value=0, max_value=20, value=0)
    parch = st.number_input('How many parents or children do you have?', min_value=0, max_value=20, value=0)
    ticket = st.text_input("What's your ticket number?", placeholder='Type Here ...')
    fare = st.number_input('How much is your ticket?', min_value=0., max_value=600., value=0.)
    cabin = st.text_input("What's your cabin number?", placeholder='Type Here ...')
    embarked = st.selectbox('Choose your embarkation port:', options=["Cherbourg", "Queenstown", "Southampton"])[0]
    click = st.button('Predict')
    
    if (click):
        prediction = predict_manual()
        st.subheader('Result:')
        col1, col2, col3 = st.columns([1,1,1])
        
        with col2:
            if prediction == 1:
                st.success('Yes, we could survive! :heartpulse:')
            else:
                st.error('No, we may meet Jack :angel:')
    
else:
    click=False
    test_data = st.file_uploader("Upload your input file", type=["csv"])
    if (test_data is not None):
        test_feature = pd.read_csv(test_data)
        test_index = test_feature['PassengerId']
        test_feature = test_feature.set_index('PassengerId')
        click = st.button('Predict')
        
        if (click):
            res = predict_file()
            st.subheader('Result:')
            
            col1, col2, col3 = st.columns([1,1,1])
            
            with col2:
                st.write(res)
        



