import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
#Load ML model
m1 = tf.keras.models.load_model("abc")
labels = np.array(['A','A+','B','C','D','E','F'])

st.title("Grading Application")


with st.form("my_form"):
   st.write("Grading Application")
   s1 = st.number_input("Subjec1:")
   s2 = st.number_input("Subjec2:")
   s3 = st.number_input("Subjec3:")
   s4 = st.number_input("Subjec4:")
   s5 = st.number_input("Subjec5:")
   # Every form must have a submit button.
   submitted = st.form_submit_button("Predict Grade")
   
   if submitted:
       st.write("Total Marsk" , 500, "Obtained Marks", s1+s2+s3+s4+s5)
       x = [[s1,s2,s3,s4,s5]]# input
       output = labels.take(np.argmax(m1.predict(x), axis=1))
       st.write("Predicted Grade:",output)

st.write("Outside the form")