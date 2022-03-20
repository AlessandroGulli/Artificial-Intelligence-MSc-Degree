from flask_sqlalchemy import SQLAlchemy
from web import app
from web import db
from datetime import datetime


class Sensorfeed(db.Model):
    id            =  db.Column('sensor_id', db.Integer, primary_key = True)
    mac_address   =  db.Column(db.String(100))
    temperature   =  db.Column(db.String(100))
    air_humidity  =  db.Column(db.String(100))
    soil_humidity =  db.Column(db.String(100))
    water_level   =  db.Column(db.String(100))
    ndvi          =  db.Column(db.String(100))
    timestamp     =  db.Column(db.Integer)

    def __init__(self, mac_address, temperature, air_humidity, soil_humidity, water_level, ndvi, time):
        self.mac_address   = mac_address
        self.temperature   = temperature
        self.air_humidity  = air_humidity
        self.soil_humidity = soil_humidity
        self.water_level   = water_level
        self.ndvi          = ndvi
        self.timestamp     = time

class PlantInfofeed(db.Model):

    __bind_key__ = 'plants_info'

    id                      =  db.Column('info_id', db.Integer, primary_key = True)
    plant_name              =  db.Column(db.String(100))
    mac_address             =  db.Column(db.String(100))  
    latitude                =  db.Column(db.String(100))
    longitude               =  db.Column(db.String(100))
    outdoor                 =  db.Column(db.Boolean)
    soil_hysteresis_high    =  db.Column(db.String(100))
    soil_hysteresis_low     =  db.Column(db.String(100))
    water_level_alarm       =  db.Column(db.String(100))

    def __init__(self, plant_name, mac_address, latitude, longitude, outdoor, soil_hysteresis_high, soil_hysteresis_low, water_level_alarm):
        self.plant_name            = plant_name
        self.mac_address           = mac_address
        self.latitude              = latitude
        self.longitude             = longitude
        self.outdoor               = outdoor
        self.soil_hysteresis_high  = soil_hysteresis_high
        self.soil_hysteresis_low   = soil_hysteresis_low
        self.water_level_alarm     = water_level_alarm


