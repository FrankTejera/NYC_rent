#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:06:17 2022

@author: ftejera
"""

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
import seaborn as sns
sns.set()

st.write("""
# NYC Rental Price Prediction

This app predicts the **_NYC Rental Price_**

Frank Tejera 

""")
st.write('---')



import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('image_file.png')  

# Loads the Boston House Price Dataset
# boston = datasets.load_boston()
# X = pd.DataFrame(boston.data, columns=boston.feature_names)
# Y = pd.DataFrame(boston.target, columns=["MEDV"])


features_to_use = [
         "rooms", "beds", "baths","neighborhood","bike room",
            "concierge",
            "deck", "doorman", "furnished", "garage parking",
            "green building", "gym", "garden", "laundry in building", "live-in super",
            "package room", "parking available", "roof deck",
            "storage available", "terrace", "virtual doorman"]

df_3 = pd.read_csv('df_temp.csv')

df_1 = pd.DataFrame({'Number': [0, 1, 2, 3, 4, 5,6]})
df_2 = pd.DataFrame({'YesNo': ['No','Yes']})

convertion = {'Yes': 1, 
              'No': 0}


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    neighborhood = st.sidebar.selectbox('Neighborhood', df_3['neighborhood'])
    rooms = st.sidebar.selectbox('Rooms', df_1['Number'])
    beds = st.sidebar.selectbox('Beds', df_1['Number'])
    baths = st.sidebar.selectbox('Baths', df_1['Number'])
    gym = st.sidebar.selectbox('Gym', df_2['YesNo'])
    deck = st.sidebar.selectbox('Deck', df_2['YesNo'])
    roof_deck = st.sidebar.selectbox('Roof Deck', df_2['YesNo'])
    garden = st.sidebar.selectbox('Garden', df_2['YesNo'])
    storage = st.sidebar.selectbox('Storage available', df_2['YesNo'])
    laundry = st.sidebar.selectbox('Laundry in Building', df_2['YesNo'])
    garage = st.sidebar.selectbox('Garage Parking', df_2['YesNo'])
    bike_room = st.sidebar.selectbox('Bike Room', df_2['YesNo'])
    doorman = st.sidebar.selectbox('Doorman', df_2['YesNo'])
    terrace = st.sidebar.selectbox('Terrace', df_2['YesNo'])
    super_live = st.sidebar.selectbox('Live-in Super', df_2['YesNo'])
    parking = st.sidebar.selectbox('Parking Available', df_2['YesNo'])

    data = {'neighborhood': neighborhood,
            'rooms': rooms,
            'beds': beds,
            'baths': baths,
            'bike room': convertion[bike_room],
            'deck': convertion[deck],
            'doorman': convertion[doorman],
            'garage parking': convertion[garage],
            'gym': convertion[gym],
            'garden': convertion[garden],
            'laundry in building': convertion[laundry],
            'live-in super': convertion[super_live],
            'parking available': convertion[parking],
            'roof deck': convertion[roof_deck],
            'storage available': convertion[storage],
            'terrace': convertion[terrace],
            
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.subheader('Specified Input Parameters')
st.write(df)
st.write('---')

# # Build Regression Model
# model = RandomForestRegressor()
# model.fit(X, Y)
# # Apply Model to Make Prediction
# prediction = model.predict(df)

#Load the trained model. (Pickle file)
model = pickle.load(open('pipe_webapp.pkl', 'rb'))

prediction = model.predict(df)

st.subheader('Estimate Baseline Rent Price')

resutl = st.button('Prediction')

if resutl:
    st.write('The prediction for the rental: ', int(prediction))
    st.write('---')
    
else:
    st.write('The prediction for the rental: ')
    st.write('---')

# Load the data set 
df_temp = pd.read_csv('df_temp.csv')
features_to_use = [
         "rooms", "beds", "baths","neighborhood","bike room", "board approval required", "cats and dogs allowed", "central air conditioning",
            "concierge", "cold storage", "community recreation facilities", "children's playroom",
            "deck", "dishwasher", "doorman", "elevator", "full-time doorman", "furnished", "garage parking",
            "green building", "gym", "garden", "guarantors accepted", "laundry in building", "live-in super",
            "package room", "parking available", "patio", "pets allowed", "roof deck", "smoke-free",
            "storage available", "sublet", "terrace", "virtual doorman", "washer/dryer in-unit", "waterview",
            "waterfront"]

X = df_temp[features_to_use].fillna(0)
y = df_temp['price']

# Find similar apartments
temp = df['neighborhood'][0]
temp_1 = df['beds'][0]

links = df_temp[(df_temp.neighborhood == temp) 
                & (df_temp.price < prediction[0] + 50)]
links = links.sort_values('price',ascending=False)
links = links['link']

# st.header('Simmilar appartments')
# st.write(links[:5])
# st.write('---')


col1, col2 = st.columns((1,2))

col2.subheader("Prediction vs Actual Price")
col1.subheader("Model")

# Get the real model for some plotting 
model_real = pickle.load(open('pipe.pkl', 'rb'))
predictions = model_real.predict(X)


with col1:
#    st.header('Model')
    # errors = abs(predictions - y)
    # accuracies = []
    # st.write('Mean Absolute Error:', round(np.mean(errors), 2))
    
    # mape = 100 * (errors / y)
    # accuracy = 100 - np.mean(mape)
    # st.write('Accuracy:', round(accuracy, 2), '%.')
    st.write('**Random Forest Regression**')
    st.write('_R-squared_:', round(0.713, 2))
    st.write('---')
    st.write('**Data**')
    st.write('Rentals: ',int(12728))
    st.write('Features: ', int(221))
    st.write('NYC rental information:')          
    """
    * Bike room
    * Board approval required
    * Cats and dogs allowed
    * Community recreation facilities
    ...
    """
    
with col2:
    N = 50
    colors = np.random.rand(N)
    area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
    
    x_scatter = model_real.predict(X)
    y_scatter = y
    
    x_plot = []
    y_plot = []
    for i in range(len(x_scatter)):
        if x_scatter[i] != y_scatter[i]:
            x_plot.append(x_scatter[i])
            y_plot.append(y_scatter[i])
    
    fig, ax = plt.subplots() 
#    plt.title('Model prediction vs Actual price')
    plt.scatter(x_plot,y_plot,s=1, alpha=0.5)
    plt.xlim([0,10000])
    plt.ylim([0,10000])
    plt.xlabel('Value ($)')
    plt.ylabel('Prediction ($)') 
    plt.figure(figsize=(10,10))
    fig.set_size_inches(4, 4, forward=True)
    
    st.pyplot(fig)
  

st.write('---')
st.subheader('Similar apartments')
for ele in links[:5]:
    st.write(ele)
    

st.write('---')
st.subheader('Feature Importance')

# Plot of the importance
factor_importance = model['randomForest'].feature_importances_

indx = np.argsort(-factor_importance)

factor_name_10 = ['rooms',
 'N: Hudson Square',
 'beds',
 'N: Elmhurst',
 'deck',
 'storage available',
 'parking available',
 'garage parking',
 'laundry in building',
 'doorman',
 'roof deck',
 'live-in super',
 'gym',
 'garden',
 'N: Hunters Point',
 'N: Bedford-Stuyvesant',
 'bike room',
 'N: Lincoln Square',
 'baths',
 'terrace']

foctor_importance_10 = []

for el in indx:
    foctor_importance_10.append(factor_importance[el])
    
# The values
y = foctor_importance_10[:20]

# Color code
my_cmap = plt.get_cmap("plasma")
val = np.linspace(0,1,20)

font = {
        'weight': 'normal',
        'size': 18,
        }

font = {
        'weight': 'normal',
        'size': 20,
        }

font_title = {
        'weight': 'bold',
        'size': 20,
        }

N = 20

import matplotlib.pyplot as plt
y = foctor_importance_10[:20]

x_ticks = [0, 0.02, 0.04, 0.08, 0.10, 0.12, 0.14]

# Color code
my_cmap = plt.get_cmap("plasma")
val = np.linspace(0,1,20)

fig, ax = plt.subplots() 
width = 0.8
ind = np.arange(len(y)) 
ax.barh(ind, y,width, color=my_cmap(val))
ax.set_yticks(ind+width/10)

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks, fontsize=20)

ax.set_yticklabels(factor_name_10[:20], minor=False,fontdict= font)
# plt.title('Feature importance in Random Forest Regression',fontdict=font_title)
plt.xlabel('Relative importance',fontdict=font_title)
plt.ylabel('Feature',fontdict=font_title) 
plt.figure(figsize=(10,8.5))
fig.set_size_inches(10, 8.5, forward=True)

st.pyplot(fig)
# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')














