import json
from pymongo import MongoClient
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt

course_cluster='your_connection_string'
course_client=MongoClient(course_cluster)

db=course_client['sample_weatherdata']
weather_data=db['data']

query = {  
    'pressure.value': { '$lt': 9999 },  
    'airTemperature.value': { '$lt': 9999 }, 
    'wind.speed.rate': { '$lt': 500 }, 
}

l=list(weather_data.find(query).limit(1000)) 
    
pressures=[x['pressure']['value'] for x in l] 
air_temps=[x['airTemperature']['value'] for x in l] 
wind_speeds=[x['wind']['speed']['rate'] for x in l] 
   
plt.clf() 
fig=plt.figure() 
  
ax=fig.add_subplot(111, projection = '3d') 
ax.scatter(pressures, air_temps, wind_speeds) 
  
ax.set_xlabel("Pressure") 
ax.set_ylabel("Air Temperature") 
ax.set_zlabel("Wind Speed") 
  
plt.show() 