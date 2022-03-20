import numpy as np
from datetime import datetime,timedelta, date
import pandas as pd
import json, requests
from web.twilio import send_wa_message, client


class Decision_Maker():

    def __init__(self):    
        self.plant_status = {}

    def process_data(self, mac_address, local_temperature, soil_humidity_param, soil_humidity, water_level_param, water_level, raining, cold_weather, outdoor, ndvi, plant_name):

        if mac_address not in self.plant_status:
            self.plant_status[mac_address] = {}
            self.plant_status[mac_address]['water_pump_on'] = False
            self.plant_status[mac_address]['Water_Level_Alert'] = False
            self.plant_status[mac_address]['NDVI_Alert'] = False
        
        #Pump Actuation Conditions
        if outdoor == True:
            if ((cold_weather == False) and (raining == False)):
                if water_level < water_level_param and local_temperature > 0:
                    if soil_humidity < soil_humidity_param[1]:  
                        if self.plant_status[mac_address]['water_pump_on'] == False:                      
                            self.plant_status[mac_address]['water_pump_on'] = True
                    elif soil_humidity > soil_humidity_param[0]:
                        if self.plant_status[mac_address]['water_pump_on'] == True:
                            self.plant_status[mac_address]['water_pump_on'] = False
                else:
                    self.plant_status[mac_address]['water_pump_on'] = False                   
            else:
                self.plant_status[mac_address]['water_pump_on'] = False 
        else:
            if water_level < water_level_param and local_temperature > 0:
                if soil_humidity < soil_humidity_param[1]:  
                    if self.plant_status[mac_address]['water_pump_on'] == False:  
                        self.plant_status[mac_address]['water_pump_on'] = True
                elif soil_humidity > soil_humidity_param[0]: 
                    if self.plant_status[mac_address]['water_pump_on'] == True:
                        self.plant_status[mac_address]['water_pump_on'] = False                   
            else:
                self.plant_status[mac_address]['water_pump_on'] = False 

        #Water Level Alert
        if water_level > water_level_param:
            if self.plant_status[mac_address]['Water_Level_Alert'] == False:
                text = 'Plant: ' + plant_name + '\nWater level is critical: no further actions will be taken \nRefill needed'
                send_wa_message(client, text)
                self.plant_status[mac_address]['Water_Level_Alert'] = True
                print('Alarm --> '+ str(mac_address)) 
        else:
            if self.plant_status[mac_address]['Water_Level_Alert'] == True:
                self.plant_status[mac_address]['Water_Level_Alert'] = False                  
        
        #NDVI Alert
        if ndvi <= 0:
            if self.plant_status[mac_address]['NDVI_Alert'] == False:
                text = 'Plant: ' + plant_name + '\nNDVI is below threshold \nHealth check needed'
                send_wa_message(client, text)
                self.plant_status[mac_address]['NDVI_Alert'] = True
        else:
            self.plant_status[mac_address]['NDVI_Alert'] = False

        #print(self.plant_status)
        return 

    def get_actuator_status(self, mac_address):
        if mac_address not in self.plant_status:            
            return "Not Present"
        else:    
            return self.plant_status[mac_address]


def get_weather_info(API_KEY, lat,lon,forecast_horizon_temperatures, forecast_horizon_weather, icing_temperature):
    #url = 'https://api.openweathermap.org/data/2.5/weather?id={}&units={}&appid={}'.format(city_id,units,API_KEY) 
    url ='https://api.openweathermap.org/data/2.5/onecall?lat={}&lon={}&exclude={}&appid={}'.format(lat,lon,'current,minutely,daily,alerts',API_KEY)
    myGET = requests.get(url)
    responseJsonBody= myGET.json()

    avg_temp = 0
    raining_vect = []
    raining = False
    cold_weather = False
    for index, item in enumerate(responseJsonBody['hourly']):
        if index == forecast_horizon_weather:
            #print(item['dt'])        
            #print(item['temp'])
            #print(item['weather'][0]['main'])
            #print(item['weather'][0]['description']) 
            if 'Rain' or 'Light Rain' in raining_vect:
                raining = True                                
        if index < forecast_horizon_temperatures:
            avg_temp += int(item['temp'])
            raining_vect.append(item['weather'][0]['main'])
        else:
            break

    avg_temp = round((avg_temp/forecast_horizon_temperatures) - 273.15, 3)

    if avg_temp < icing_temperature:
        cold_weather = True

    return raining,cold_weather

