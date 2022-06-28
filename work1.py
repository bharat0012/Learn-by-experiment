#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[3]:


st.title("DEMO OF Random Forest Classifier MODEL")


# In[4]:


st.write("ANALYSIS OF Random Forest Classifier ON IRIS DATASET")


# In[5]:


iris_data = load_iris()


# In[6]:


features = pd.DataFrame(
    iris_data.data, columns=iris_data.feature_names
)
target = pd.Series(iris_data.target)


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target
)


# In[8]:


class StreamlitApp:

    def __init__(self):
        self.model = RandomForestClassifier()

    def train_data(self):
        self.model.fit(x_train, y_train)
        return self.model

    def construct_sidebar(self):

        cols = [col for col in features.columns]

        st.sidebar.markdown(
            '<p class="header-style">Iris Data Classification</p>',
            unsafe_allow_html=True
        )
        sepal_length = st.sidebar.selectbox(
            f"Select {cols[0]}",
            sorted(features[cols[0]].unique())
        )

        sepal_width = st.sidebar.selectbox(
            f"Select {cols[1]}",
            sorted(features[cols[1]].unique())
        )

        petal_length = st.sidebar.selectbox(
            f"Select {cols[2]}",
            sorted(features[cols[2]].unique())
        )

        petal_width = st.sidebar.selectbox(
            f"Select {cols[3]}",
            sorted(features[cols[3]].unique())
        )
        values = [sepal_length, sepal_width, petal_length, petal_width]

        return values


    def construct_app(self):

        self.train_data()
        values = self.construct_sidebar()

        values_to_predict = np.array(values).reshape(1, -1)

        prediction = self.model.predict(values_to_predict)
        prediction_str = iris_data.target_names[prediction[0]]
        probabilities = self.model.predict_proba(values_to_predict)

        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="header-style"> Iris Data Predictions </p>',
            unsafe_allow_html=True
        )

        column_1, column_2 = st.beta_columns(2)
        column_1.markdown(
            f'<p class="font-style" >Prediction </p>',
            unsafe_allow_html=True
        )
        column_1.write(f"{prediction_str}")

        column_2.markdown(
            '<p class="font-style" >Probability </p>',
            unsafe_allow_html=True
        )
        column_2.write(f"{probabilities[0][prediction[0]]}")

        
        st.markdown(
            '<p class="font-style" >Probability Distribution</p>',
            unsafe_allow_html=True
        )
      

        return self


# In[9]:


sa = StreamlitApp()
sa.construct_app()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




