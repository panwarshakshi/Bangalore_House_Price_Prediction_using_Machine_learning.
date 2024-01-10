#pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
loaded_model=1
#loaded_model = joblib.load('./model.pkl.load')
data_frame = pickle.load(open('used_dataframe.p', 'rb'))
df = df = pd.read_csv('Bengaluru_House_Data.csv')
loaded_model= pickle.load(open('model.pkl','rb'))

def ret_cols():
    return df['location'].unique()

#print(df['location'].unique())
# Create a function to predict house prices
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(data_frame.columns==location)[0][0]

    X = np.zeros(len(data_frame.columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1

    return loaded_model.predict([X])[0]

# Set the page title and icon
st.set_page_config(page_title='Bangalore House Price Predictor', page_icon=':house:')

# Add an image to the page
st.image('SHAKSHI.jpg', use_column_width=True)

# Add a title to the page
st.title('Bangalore House Price Predictor')

cols = ret_cols()

# Create input widgets for the user to input data

location = st.selectbox('Select Location', cols)
sqft = st.slider('Square Feet', 100, 10000, 1000)
bath = st.slider('Number of Bathrooms', 1, 50, 2)
bhk = st.slider('Number of Bedrooms', 1, 20, 2)

result = predict_price(location, sqft, bath, bhk)

if st.button('Predict Price'):
    result = predict_price(location, sqft, bath, bhk)
    st.success('The estimated house price is Rs. {:.2f} lakhs'.format(result))


