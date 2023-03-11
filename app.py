import streamlit as st
import pickle

temp = st.number_input('Input temperature', value=20)
if st.button('Predict'):
  st.caption('Predict Revenue')
  model = pickle.load(open('model.pickle', "rb"))
  rev = model.predict([[temp]])
  st.success(f'{rev[0]}')
