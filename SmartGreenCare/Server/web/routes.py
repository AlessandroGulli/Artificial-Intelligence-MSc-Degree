from altair import Chart, X, Y, Axis, Data, DataFormat
import pandas as pd
import numpy as np
from flask import render_template, url_for, flash, redirect, request, make_response, jsonify, abort
from web import app
from web import db
from web.utils import utils
from web.utils.utils import Decision_Maker
from web.db import Sensorfeed,PlantInfofeed
import json, time, arrow, pytz
from datetime import datetime, timezone
import csv, os

manager = Decision_Maker()


@app.errorhandler(404)
def page_not_found(error):
    return 'Error', 404

@app.errorhandler(502)
def bad_gateway(error):
    return 'Error', 502

@app.errorhandler(500)
def internal_server_error(error):
    return 'Error', 500

@app.route("/")
@app.route("/main")
def plot_main():

    plants_names = []
    mac_addresses = []
    records=PlantInfofeed.query.order_by(PlantInfofeed.id.desc()).all()
 
    for record in records:
        plants_names.append(record.plant_name)
        mac_addresses.append(record.mac_address)
    
    plant_name = plants_names[0]
    mac_address = mac_addresses[0]

    records=Sensorfeed.query.order_by(Sensorfeed.id.desc()).filter_by(mac_address=mac_address).all()

    temperature   = []
    air_humidity  = []
    soil_humidity = []
    water_level   = []
    ndvi          = []

    for record in records:
        recordObject = {
            'mac_address'  : record.mac_address,
            'temperature'  : record.temperature,
            'air_humidity' : record.air_humidity,
            'soil_humidity': record.soil_humidity,
            'water_level'  : record.water_level,
            'ndvi'         : record.ndvi,
            'date_time'    : record.timestamp
        }

        temperature.append([int(record.timestamp),float(record.temperature)])
        air_humidity.append([int(record.timestamp),float(record.air_humidity)])
        soil_humidity.append([int(record.timestamp),float(record.soil_humidity)])
        water_level.append([int(record.timestamp),float(record.water_level)])
        ndvi.append([int(record.timestamp),float(record.ndvi)])

    records = PlantInfofeed.query.filter_by(mac_address=mac_address).limit(1).all()

    if len(records) > 0:
        for record in records:
            outdoor = record.outdoor
            soil_hysteresis_high = record.soil_hysteresis_high
            soil_hysteresis_low = record.soil_hysteresis_low
            water_level_alarm = record.water_level_alarm

    context = {"plant_name": plant_name, "mac_address":mac_address,"plants_names": plants_names,
               "temperature": temperature[::-1], "air_humidity": air_humidity[::-1],
               "soil_humidity": soil_humidity[::-1], "water_level": water_level[::-1], "ndvi": ndvi[::-1],
                "outdoor": outdoor, "soil_hysteresis_high": soil_hysteresis_high, "soil_hysteresis_low": soil_hysteresis_low, "water_level_alarm": water_level_alarm}

    return render_template('main.html', context=context)

@app.route("/plant_dashboard", methods=['POST'])
def plant_dashboard():

    plant_name = request.form['plant_name']

    plants_names = []
    records=PlantInfofeed.query.order_by(PlantInfofeed.id.desc()).all()
    for record in records:
        plants_names.append(record.plant_name)
        if plant_name == record.plant_name:
            db_entry = record.mac_address

    records=Sensorfeed.query.order_by(Sensorfeed.id.desc()).filter_by(mac_address=db_entry).all()

    temperature   = []
    air_humidity  = []
    soil_humidity = []
    water_level   = []
    ndvi          = []

    for record in records:
        recordObject = {
            'mac_address'  : record.mac_address,
            'temperature'  : record.temperature,
            'air_humidity' : record.air_humidity,
            'soil_humidity': record.soil_humidity,
            'water_level'  : record.water_level,
            'ndvi'         : record.ndvi,
            'date_time'    : record.timestamp
        }
        
        temperature.append([int(record.timestamp),float(record.temperature)])
        air_humidity.append([int(record.timestamp),float(record.air_humidity)])
        soil_humidity.append([int(record.timestamp),float(record.soil_humidity)])
        water_level.append([int(record.timestamp),float(record.water_level)])
        ndvi.append([int(record.timestamp),float(record.ndvi)])

    records = PlantInfofeed.query.filter_by(mac_address=db_entry).limit(1).all()

    if len(records) > 0:
        for record in records:
            outdoor = record.outdoor
            soil_hysteresis_high = record.soil_hysteresis_high
            soil_hysteresis_low = record.soil_hysteresis_low
            water_level_alarm = record.water_level_alarm

    context = {"plant_name": plant_name, "mac_address":db_entry,"plants_names": plants_names,
               "temperature": temperature[::-1], "air_humidity": air_humidity[::-1],
               "soil_humidity": soil_humidity[::-1], "water_level": water_level[::-1], "ndvi": ndvi[::-1],
                "outdoor": outdoor, "soil_hysteresis_high": soil_hysteresis_high, "soil_hysteresis_low": soil_hysteresis_low, "water_level_alarm": water_level_alarm}


    return render_template('item.html', context=context)



@app.route('/manage_actuators/<mac_address>', methods=['GET'])
def manage_actuators(mac_address):

    response = make_response(json.dumps(manager.get_actuator_status(mac_address)))
    response.content_type = 'application/json'

    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'

    return response

@app.route('/manage_sensors', methods=['POST'])
def manage_sensors():

    
    time_ref = datetime(datetime.now().year, datetime.now().month,datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second)
    time_ref = (arrow.get(time_ref).timestamp)*1000
    req_data = request.values
    sf = Sensorfeed(req_data['mac_address'], req_data['temperature'], req_data['air_humidity'], req_data['soil_humidity'], req_data['water_level'], req_data['ndvi'], time_ref)

    db.session.add(sf)
    db.session.commit()

    return str(sf.id)

@app.route('/last_data/<mac_address>', methods=['GET'])
def last_data(mac_address):

    data = []

    records = Sensorfeed.query.order_by(Sensorfeed.id.desc()).filter_by(mac_address=mac_address).limit(1).all()    
    for record in records:
        data = [float(record.temperature), float(record.air_humidity), float(record.soil_humidity), float(record.water_level), float(record.ndvi)] 
    
    records = PlantInfofeed.query.filter_by(mac_address=mac_address).limit(1).all()
    for record in records:
        data.append(float(record.soil_hysteresis_high))
        data.append(float(record.soil_hysteresis_low))
        data.append(record.plant_name)
        data.append(record.outdoor)
        data.append(float(record.water_level_alarm))

    response = make_response(json.dumps(data))
    response.content_type = 'application/json'

    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'

    return response

@app.route('/qr_code', methods=['GET','POST'])
def qr_code():

    if request.method == "POST":        
        req_data = request.get_json()

        if req_data is None:
            print("Message Text not Found")
            statement = "Protocol Error"
        else:   

            records = PlantInfofeed.query.filter_by(mac_address=req_data['code']).limit(1).all()

            if len(records) > 0:
                for record in records:
                    if  (record.soil_hysteresis_high == req_data['soil_hysteresis_high']) and (record.soil_hysteresis_low == req_data['soil_hysteresis_low']\
                        and (record.water_level_alarm == req_data['water_level_alarm']) and (record.outdoor == req_data['outdoor'])):
                        statement = "Sensor Parameters Values Already Present, Nothing Due" 
                    else:
                        record.plant_name           = req_data['name']
                        record.outdoor              = req_data['outdoor']
                        record.soil_hysteresis_high = req_data['soil_hysteresis_high']
                        record.soil_hysteresis_low  = req_data['soil_hysteresis_low']  
                        record.water_level_alarm    = req_data['water_level_alarm'] 
                        record.latitude             = req_data['lat']
                        record.longitude            = req_data['long']   
                           
                        db.session.commit()
                        statement = "New Sensor Parameters Values Updated" 
            else:
                pif = PlantInfofeed(req_data['name'], req_data['code'], req_data['lat'], req_data['long'], req_data['outdoor'], req_data['soil_hysteresis_high'], req_data['soil_hysteresis_low'], req_data['water_level_alarm'])
                db.session.add(pif)
                db.session.commit()
                statement = "New Plant Correctly Added"
 
    elif request.method == "GET":

        records=PlantInfofeed.query.all()

        data = []
        for record in records:
            data.append([record.plant_name,record.mac_address])

        response = make_response(json.dumps(data))
        response.content_type = 'application/json' 

        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers['Cache-Control'] = 'public, max-age=0'
        statement = response

    return statement

@app.route('/update_param', methods=['POST'])
def update_param():
    if request.method == "POST":
        req_data = request.form
        
        if req_data is None:
            print("Message Text not Found")
            statement = "Protocol Error"
        else:

            records = PlantInfofeed.query.filter_by(mac_address=req_data['code']).limit(1).all()            

            if len(records) > 0:
                for record in records:
                    if req_data['outdoor'] == 'false':
                        outdoor = False
                    else:
                        outdoor = True
                    if req_data['soil_hysteresis_high'] == '':
                        soil_hysteresis_high = record.soil_hysteresis_high
                    else:
                        soil_hysteresis_high = req_data['soil_hysteresis_high']
                    if req_data['soil_hysteresis_low'] == '':
                        soil_hysteresis_low = record.soil_hysteresis_low
                    else:
                        soil_hysteresis_low = req_data['soil_hysteresis_low']
                    if req_data['water_level_alarm'] == '': 
                        water_level_alarm = record.water_level_alarm
                    else:
                        water_level_alarm = req_data['water_level_alarm']    

                    if (record.soil_hysteresis_high == soil_hysteresis_high) and (
                            record.soil_hysteresis_low == soil_hysteresis_low) \
                            and (record.water_level_alarm == water_level_alarm) and (
                                    record.outdoor == outdoor):
                        statement = "Sensor Parameters Values Already Present, Nothing Due"
                    else:
                        record.plant_name = req_data['name']
                        record.outdoor = outdoor
                        record.soil_hysteresis_high = soil_hysteresis_high
                        record.soil_hysteresis_low = soil_hysteresis_low
                        record.water_level_alarm = water_level_alarm

                        db.session.commit()
                        statement = "New Sensor Parameters Values Updated"
            else:
                statement="Values not inserted"

    return statement
