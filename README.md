# flightdataanomaly
Anomaly detection on a Flight dataset

Using unsupervised learning algorithm, I will find for anomalies on a Flight dataset

this project will be create on Jupyter Notebook

tools used

Numpy - Matplotlib - pandas - seaborn - tensorflow

Numpy -> to make math 
Pandas -> Data edit
Matplotlib -> Data visualization
Seaborn -> Data visialization based on Matplotlib
Scikit-learn  -> 

#Install libraries

!pip install numpy
!pip install tensorflow
!pip install matplotlib
!pip install pandas
!pip install seaborn


#Import Libraries

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# upload dataset https://www.kaggle.com/datasets/mahoora00135/flights?resource=download

from google.colab import files
uploaded = files.uploaded ()


# Load dataset using pandas

df = pd.read_csv('flights.csv')

#show first line of the dataframe

df.head()

#show dataframe info

df.info

#show dataframe stats 

df.describe()

#made a histogram to show the array delay

plt.figure(figsize =(10, 6))
sns.histplot(df['arr_delay'], bins=30, kde = True)
plt.title('Array Delay Distribution')
plt.xlabel('Array Delay on min')
plt.ylabel('Frecuency')
plt.show()


#look for the available features
print(df.columns)

#pick related data to the analysis and normalize de data

features = ['month', 'origin', 'dest', 'arr_delay', 'dep_delay', 'distance', 'air_time']

#select the new columns and delete de previus columns

df.selected = df[features].dropna()

#show the new Dataframe

#df_selected.head()


# encode one-hot for the columns origin and dest to made as numbers instead of names and verify

df_encoded = pd.get_dummies(df_selected, columns = ['origin', 'dest'])

df_encoded.head()

#convert the dataframe to a tensorflow

numeric_features = ['arr_delay', 'dep_delay', 'distance', 'air_time', 'month']

data_tensor = tf.constant(df_encoded[numeric_features].values, dtype=tf.float32)

#calculate the mean and standard deviation

mean = tf.reduce_mean(data_tensor, axis = 0)
std_dev = tf.math.reduce_std(data_tensor, axis = 0)

#normalize the data
normalized_data = (data_tensor - mean) / std_dev
