from web import app
from datetime import datetime
from flask_apscheduler import APScheduler
from web.db import Sensorfeed,PlantInfofeed
from web.routes import manager
from web.utils.utils import get_weather_info
import requests
import numpy as np

API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

#lat = '44.886768'
#lon = '11.0662'

forecast_horizon_weather      = 2  #hours
forecast_horizon_temperatures = 10 #degrees
icing_temperature             = 0 #degrees

def my_job(text):

    records_plantinfo = PlantInfofeed.query.order_by(PlantInfofeed.id.desc()).all()
    for record_plantinfo in records_plantinfo:
        records_sensorfeed = Sensorfeed.query.order_by(Sensorfeed.id.desc()).filter_by(mac_address=record_plantinfo.mac_address).limit(1).all()
        #raining,cold_weather = get_weather_info(API_KEY,lat,lon,forecast_horizon_temperatures, forecast_horizon_weather, icing_temperature)
        if record_plantinfo.outdoor == True:
            raining, cold_weather = get_weather_info(API_KEY,record_plantinfo.latitude,record_plantinfo.longitude,forecast_horizon_temperatures, forecast_horizon_weather, icing_temperature)
            print("Node: "+ record_plantinfo.mac_address + " Raining: " + str(raining) + " Cold: " + str(cold_weather))
        else:
            raining, cold_weather = False, False
        for record_sensorfeed in records_sensorfeed:
            manager.process_data(record_plantinfo.mac_address, float(record_sensorfeed.temperature), [float(record_plantinfo.soil_hysteresis_low),float(record_plantinfo.soil_hysteresis_high)], \
                                 float(record_sensorfeed.soil_humidity), float(record_plantinfo.water_level_alarm), float(record_sensorfeed.water_level), raining, cold_weather, record_plantinfo.outdoor, \
                                float(record_sensorfeed.ndvi), record_plantinfo.plant_name) 

scheduler = APScheduler()