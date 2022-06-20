intro = '''
# Airline Satisfaction Analysis
### Prediction of passenger satisfaction
##### Context and Content:
What factors are highly correlated to a satisfied (or dissatisfied) passenger in an airline service.
#### 1. Data Source 
We found this dataset from kaggle, a series of files. This data set contains a survey on air passenger satisfaction. The dataset is free of cost and can be easily downloaded from the Kaggle website. 
#### 2. Data merging and cleaning
The next step was to merge the dataframes from Kaggle to a single dataframe and to clean it. We converted data into useful information as well as created new columns to analyize the data better.Identified the Null values and Filling in the missing values.
#### 3. Data Visualizations
In this part we did exploritory data analysis on the data and understood the relationships between the variables of the data by using graphs.   
#### 4. Applying Models
We applied four classification models, the logistic regressition, random forest, decision tree and Naive Bayes and then we compared the results. 
#### 5. Streamlit Presentation
We are presenting our project using streamlit platform. We used streamlite because streamlit is an open-source python framework for building web apps for Machine Learning and Data Science. Streamlit allows you to write an app the same way you write a python code.
'''


intro_herramientas_fuentes = '''
---
## Sources and References
[Kaggle]( https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction) </br>
[Streamlit](https://streamlit.io/) </br>
[Streamlit Doc](https://docs.streamlit.io/) </br>
[Concepts From Streamlit Gallery](https://share.streamlit.io/casiopa/eda-imdb/main/src/utils/streamlit/EDA_IMDb_main.py)


## Used tools
| Data mining		| Visualization 	|
|---				|---				|
| - Jupyter Notebook| - Streamlit		|
| - Pycharm			| - Python			|
| - Python			| - Numpy			|
| - Pandas			| - Matplotlib		|
| - Numpy			| - Seaborn		    |
| - Sklearn  		| 		            |

'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Importing data
eda_data = pd.read_csv(r'airline_passenger_satisfaction.csv')
data = pd.read_csv(r'airline_passenger_satisfaction.csv', index_col=False)
df = pd.read_csv(r'airline_passenger_satisfaction.csv')





def set_home():
    st.image("pexels-pascal-renet-113017.jpg")
    st.write(intro, unsafe_allow_html=True)
    st.write(intro_herramientas_fuentes, unsafe_allow_html=True)


def set_data():
    st.title("DataFrame:")
    st.write("This dataset designed to understand the factors that lead a person to Satisfaction or Neutral (dissatisfied). By using classification models we will predict the probability of a passenger being satisfied or neutral, as well as interpreting affected factors on passenger comfort.")
    st.write(">***21287 entries | 15 columns***")
    st.dataframe(data)
    st.write(">***Let's analyize the dataframe.***")
    st.dataframe(data.describe())
    st.text("")
    st.title("*Satisfaction Distribution:*")
    st.write("By analyizing the satisfaction distribution we can understand that the 43.4% of the airline passengers were satisfied.")
    col1, col2 = st.columns(2)
    with col1:
        total = data['Satisfaction'].value_counts()
        fig1 = total.plot.pie(shadow=True, explode=(0, 0.1), startangle=0, autopct='%1.1f%%')
        labels = ['Not Satisfied','Satisfied']
        plt.legend(labels)
        plt.axis('equal')
        st.pyplot(fig1.figure)
    with col2:
        st.markdown(' ')
    st.write(">***Now we will analyize which factors/attributes effects these passenger satisfaction decisions.***")

def set_analysis():
    st.title("*Analyizing Passengers that were Satisfied:*")
    st.write("*We are focusing on passengers who were satisfied with the airline services, lets analyze which factors effects their decission.*")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1: Satisfaction Rating provided by the various Passengers On: ")
        st.markdown('Arrival Delay, Departure and Arrival Time Convenience, Ease of Online Booking AND Check-in Service.')
        st.image("a.png")

    with col2:
        st.markdown("#### 2: Satisfaction Rating provided by the various Passengers On: ")
        st.markdown('Online Boarding, Gate Location, On-board Service AND Seat Comfort.')
        st.image("b.png")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 3: Satisfaction Rating provided by the various Passengers On: ")
        st.markdown('Leg Room Service, Cleanliness, Food and Drink AND In-flight Service.')
        st.image("c.png")

    with col2:
        st.markdown("#### 4: Satisfaction Rating provided by the Various Passengers On: ")
        st.markdown('In-flight Wifi Service, In-flight Entertainment AND Baggage Handling.')
        st.image("d.png")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 5: Satisfaction Level of Passengers On:")
        st.markdown('Cleanliness.')
        st.image("e.png")
    with col2:
        st.write("#### 6: Satisfaction Level of Passengers On:")
        st.markdown('Baggage Handling.')
        st.image("f.png")

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 7: Satisfaction Level of Passengers On:")
        st.markdown('Food and Drink.')
        st.image("g.png")

    with col2:
        st.markdown("#### 8: Satisfaction Indicator")
        st.markdown('Satisfied or Dissatisfied')
        ##st.write("*From this graph you can understand that most*")
        st.image("h.png")

    col1, col2 = st.columns(2)

def set_classmod():
    st.title("*Classification Model*")
    st.write("We applied Four classification models, logistics regression, random forest model, Decision tree and Naive bayes . They gave a mean absolute error of ***0.186*** and ***0.074*** For the models. These results can be improved further by optimizing")
    st.subheader("*Prediction Using Logistic Regression Model*")
    st.write("##### Mean absolute error:")
    st.write("0.18627592512391125")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader("*Prediction Using Random Forest*")
    st.write("##### Mean absolute error:")
    st.write("0.071680389394159")

    st.subheader("*Prediction Using Decision Tree Classifier*")
    st.write("##### Accuracy:")
    st.write("93.10324915306436")

    st.subheader("*Prediction Using Naive Bayes*")
    st.write("##### Accuracy:")
    st.write("81.68982907299045")

    






import streamlit as st
#from as_function import *

st.sidebar.image('download.jpg', width=250)
st.sidebar.header('Airline Satisfaction Analysis')
st.sidebar.markdown('Prediction of passenger satisfaction')


menu = st.sidebar.radio(
    "Menu:",
    ("Intro", "Data", "Analysis", "Classification Model"),
)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('Project Submitted By: Nikhil Kumar')
st.sidebar.write('Github Repositories:')
st.sidebar.write('[Nikhil Github Repository Link]()')

if menu == 'Intro':
    set_home()
elif menu == 'Data':
    set_data()
elif menu == 'Analysis':
    set_analysis()
elif menu == 'Classification Model':
    set_classmod()
