# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache
def get_data(filename):
	data = pd.read_csv(filename)

	return data


with header:
    st.title('Welcome to Salary Prediction project!')
    st.subheader('In this project we will predict salary!!')

with dataset:
    
    data = pd.read_csv('hiring.csv')
    

with features:
    st.header('we are using following features')
    st.subheader('1.Experience')
    st.subheader('2.Test Score')
    st.subheader('3.Interview score')


with model_training:
    st.header('Lets Train Model!!')
    st.text('Here you can choose features')
    Sel_col, disp_col = st.columns(2)
    Experience = Sel_col.slider('Enter Experience',min_value=0,max_value=10,value=0,step=1)
    test_score = Sel_col.slider('Enter test score',min_value=0,max_value=10,value=0,step=1)
    interview_score = Sel_col.slider('Enter interview_score',min_value=0,max_value=10,value=0,step=1)

    

    data['experience'].fillna(0, inplace=True)

    data['test_score'].fillna(data['test_score'].mean(), inplace=True)

    X = data.iloc[:, :3]

    #Converting words to integer values
    def convert_to_int(word):
        word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                    'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
        return word_dict[word]

    X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

    y = data.iloc[:, -1]
    
    #Splitting Training and Test Set
    #Since we have a very small dataset, we will train our model with all availabe data.

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()

    #Fitting model with trainig data
    regressor.fit(X, y)
    
    # prediction = regressor.predict(y)
    pred = np.round(regressor.predict([[Experience, test_score, interview_score]]))
    # print(pred)
    disp_col.subheader('The Predicted Salary is:')
    disp_col.write(pred)
    

st.header('we are using hiring dataset')
st.write(X.head(5))
print(pred)