# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:52:40 2023
@author: soura
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data= pd.read_csv("C:\\Users\\soura\\OneDrive\\Desktop\\World Bank data.csv")

data.isnull().sum()

data=data.dropna()
data


data.info()


# Create a new DataFrame with only the columns we need
renewable_data = data[['Country', 'Year', 'Renewable Energy Consumption (%)', 'Renewable Energy Production (%)']]

# Pivot the DataFrame to have one column for each country's data
renewable_data = renewable_data.pivot(index='Year', columns='Country')

print('renewable_data')


# Plot the data
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(renewable_data['Renewable Energy Consumption (%)'])
ax.set_title('Renewable Energy Consumption Trends by Country')
ax.set_xlabel('Year')
ax.set_ylabel('Renewable Energy Consumption (%)')
plt.show()


fig, ax = plt.subplots(figsize=(10,6))
ax.plot(renewable_data['Renewable Energy Production (%)'])
ax.set_title('Renewable Energy Production Trends by Country')
ax.set_xlabel('Year')
ax.set_ylabel('Renewable Energy Production (%)')
plt.show()


# # Using K-means

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(renewable_data)


# Cluster the data
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(normalized_data)


kmeans


# Add the cluster labels to the DataFrame
renewable_data['Cluster'] = clusters


# Plot the data with cluster labels
fig, ax = plt.subplots(figsize=(10,6))
for cluster in range(4):
    ax.plot(renewable_data[renewable_data['Cluster']==cluster]['Renewable Energy Consumption (%)'])
ax.set_title('Renewable Energy Consumption Trends by Country (Clustered)')
ax.set_xlabel('Year')
ax.set_ylabel('Renewable Energy Consumption (%)')
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
for cluster in range(4):
    ax.plot(renewable_data[renewable_data['Cluster']==cluster]['Renewable Energy Production (%)'])
ax.set_title('Renewable Energy Production Trends by Country (Clustered)')
ax.set_xlabel('Year')
ax.set_ylabel('Renewable Energy Production (%)')
plt.show()




