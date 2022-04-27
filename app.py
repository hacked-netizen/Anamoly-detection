# -*- coding: utf-8 -*-
"""
@author: UTKARSH
"""
import pickle
import streamlit as st
from PIL import Image
from pyimagesearch.features import quantify_image
import cv2
  

model = pickle.loads(open('anomaly_detector.model', "rb").read())
 
def load_image(image_file):
	img = Image.open(image_file)
	return img
 
def welcome():
    return 'welcome all'
  
def prediction(features):  
   
    preds = model.predict([features])[0]
    return preds
      
  
def main():
    
    st.title("Anamoly Detection")
      
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Detction of an anomalous image from uploaded image</h1>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html = True)
      

    uploaded_file = st.file_uploader("Choose a file",type=["png","jpg","jpeg"])
    if uploaded_file != None:
        image = cv2.imread(uploaded_file.name)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = quantify_image(hsv, bins=(3, 3, 3))
    
    result =""
      
    
    if st.button("Predict"):
        result = prediction(features)
        if result != None:
            label = "anomaly" if result == -1 else "normal"
            color = (0, 0, 255) if result == -1 else (0, 255, 0)
            st.success('The output is {}'.format(label))
            st.image(load_image(uploaded_file),width=250)
     
if __name__=='__main__':
    main()
