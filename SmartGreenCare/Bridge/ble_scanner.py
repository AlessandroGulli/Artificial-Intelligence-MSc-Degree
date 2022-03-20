from bluepy import btle
import struct, os, requests, json, time
from concurrent import futures
from crc import crc8
from sharing import loadData


global addr_var
global delegate_global
global perif_global
global arduino_time
global plant_index_time
global post_time
global get_time
global state_pump
global NDVI

global air_temp
global air_hum
global soil_hum
global water_level

debug = True

addr_var = []


def init_NDVI(NDVI, addr_var):
    for i in range(len(addr_var)):
        NDVI[addr_var[i]] = 0
    return NDVI    

def get_NDVI():
    result = loadData()
    for keys in result: 
	    NDVI[keys] = result[keys] 

def get_addresses():
    mac_addresses = []
    url = 'http://iotproject.ngrok.io/qr_code'
    myGET = requests.get(url)    
    responseJsonBody = myGET.json()
    for el in responseJsonBody:
        mac_addresses.append(el[1])
    return mac_addresses

def parser(data):    
    error = {"Protocol":None, "Air": None, "Soil": None, "Water": None}
    header = data[0]
    if header == 0xAF:
        #print("Valid Packet")
        status = data[1]
        if status != 1:
           if debug:
               print("Invalid Temperature/Humidity Value")
           error["Air"] = True
           temp_int = 0xFF
           temp_dec = 0xFF            
           hum_int  = 0xFF
           hum_dec  = 0xFF
        else:
            temp_int = data[2]
            temp_dec = data[3]            
            hum_int  = data[4]
            hum_dec  = data[5]
        temp_out = float(temp_int*100 + temp_dec)/float(100)
        air_humidity_out = float(hum_int*100 + hum_dec)/float(100)
        status = data[6]
        if status != 1:
            if debug:
                print("Invalid Humidity Soil Value")
            error["Soil"] = True
            soil_int = 0xFF
            soil_dec = 0xFF 
        else:    
            soil_int = data[7]
            soil_dec = data[8]                        
        soil_humidity_out = float(soil_int*100 + soil_dec)/float(100)
        status = data[9]
        if status != 1:
            if debug:
                print("Invalid Water Level Value")
            error["Water"] = True
            water_int = 0xFF
            water_dec = 0xFF
        else:
            water_int = data[10]
            water_dec = data[11]
        water_level_out = float(water_int*100 + water_dec)/float(100)
        eop = data[12]
        if eop != 0xDD:
            if debug:
                print("End of Packet Missing")
            error["Protocol"] = True
        if crc8(data, 14) != 0:
            if debug:
                print("Crc Not Valid")
            error["Protocol"] = True
    else:
        error["Protocol"] = True
        if debug:
            print("Invalid Packet")
    
    return temp_out, air_humidity_out, soil_humidity_out, water_level_out, error

def post_data(addr_var, temp, air_humidity, soil_humidity, water_level, ndvi):    
    mypostdata = {'mac_address':addr_var,'temperature':temp,'air_humidity':air_humidity, 'soil_humidity':soil_humidity,'water_level':water_level, 'ndvi':ndvi}

    url = 'http://iotproject.ngrok.io/manage_sensors'
    myPOST = requests.post(url, data=mypostdata)
    if debug:
        print(myPOST.json())    
    return

def get_data(perif_global, addr_var, cHandle, pump_state):
    url = 'http://iotproject.ngrok.io/manage_actuators/' + addr_var
    myGET = requests.get(url)
    if debug:
        print(myGET.text)
    responseJsonBody = myGET.json()    
    state = pump_state
    if responseJsonBody['water_pump_on'] == True:
        if state is 'OFF':
            msg = 'WaterPump'+'ON'                                 
            crc = crc8(msg.encode('utf-8'),11)                                
            data = []
            u = msg.encode('utf-8')
            for idx in range(len(u)):
                data.append(u[idx])                            
            data.append(crc)                                                                                                                           
            perif_global.writeCharacteristic(cHandle,bytes(data))  
            state = 'ON'     
    else:
        if state is 'ON':
            msg = 'WaterPump'+'OFF'                                
            crc = crc8(msg.encode('utf-8'),12)                                
            data = []
            u = msg.encode('utf-8')                            
            for idx in range(len(u)):
                data.append(u[idx])
            data.append(crc)                                
            perif_global.writeCharacteristic(cHandle,bytes(data))   
            state = 'OFF'
    return state
    

class MyDelegate(btle.DefaultDelegate):

    def __init__(self,params):
        btle.DefaultDelegate.__init__(self)

    def handleNotification(self,cHandle,data):
        global addr_var
        global delegate_global
        global arduino_time
        global post_time
        global get_time
        global state_pump
        global plant_index_time  
        global air_temp_avg
        global air_hum_avg
        global soil_hum_avg
        global water_level_avg
        
        for ii in range(len(addr_var)):            
            if delegate_global[ii]==self:
                try:                                  
                    #print("Address: "+addr_var[ii], addr_var)
                    temp, air_humidity, soil_humidity, water_level, error = parser(data)

                    if state_pump[ii] == 'ON':
                            get_data_time = 5
                            post_data_time = 10
                    elif state_pump[ii] == 'OFF':    
                            get_data_time = 60
                            post_data_time = 60
                    
                    if error["Protocol"] == None:
                        #if debug:
                            #print("Address: "+addr_var[ii] + " --> Temp:" + str(temp) + " *C, Air Hum:" + str(air_humidity) +  " %, Soil Hum:" + str(soil_humidity) +\
                            #  " %, Water Lev:" + str(water_level) + " cm" + " NDVI:" + str(NDVI[addr_var[ii]]))

                        air_hum_avg[ii]     = round(0.1*air_hum_avg[ii]  + 0.9*air_humidity,2)
                        air_temp_avg[ii]    = round(0.1*air_temp_avg[ii] + 0.9*temp,2)
                        soil_hum_avg[ii]    = round(0.1*soil_hum_avg[ii] + 0.9*soil_humidity,2)
                        water_level_avg[ii] = round(0.1*water_level_avg[ii] + 0.9*water_level,2)  
                        
                        if time.time() - post_time[ii] > post_data_time: 

                            if debug:
                                print("Address: "+addr_var[ii] + " --> Temp:" + str(air_temp_avg[ii]) + " *C, Air Hum:" + str(air_hum_avg[ii]) +  " %, Soil Hum:" + str(soil_hum_avg[ii]) +\
                                        " %, Water Lev:" + str(water_level_avg[ii]) + " cm" + " NDVI:" + str(NDVI[addr_var[ii]]))
                              
                            post_data(addr_var[ii], air_temp_avg[ii], air_hum_avg[ii], soil_hum_avg[ii], water_level_avg[ii], NDVI[addr_var[ii]])  
                            post_time[ii] = time.time() 
                        
                        if time.time() - arduino_time[ii] > get_data_time:    
                            state_pump[ii] = get_data(perif_global[ii], addr_var[ii], cHandle, state_pump[ii])  
                            arduino_time[ii] = time.time()                             

                        if time.time() - plant_index_time[ii] > (3600*24):
                            get_NDVI()
                            plant_index_time[ii] = time.time()  
                    return
                except:                    
                    return
    
def perif_loop(perif,indx):       
    size = len(get_addresses())
    while True:        
            try:               
                if perif.waitForNotifications(1.0):                                        
                    continue                            
            except:
                try:
                    perif.disconnect()
                except:
                    pass                
                print("disconnecting perif: "+perif.addr+", index: "+str(indx))
                reestablish_connection(perif,perif.addr,indx)
        


def reestablish_connection(perif,addr,indx):
    while True:
        try:
            print("trying to reconnect with "+addr)
            perif.connect(addr)
            print("re-connected to "+addr+", index = "+str(indx))
            return
        except:
            continue

def establish_connection(addr):
    global delegate_global
    global perif_global
    global addr_var
    global arduino_time
    global post_time    
    global get_time
    
    while True:        
        try:
            for jj in range(len(addr_var)):
                if addr_var[jj]==addr:
                    print("Attempting to connect with "+addr+" at index: "+str(jj))
                    p = btle.Peripheral(addr)
                    perif_global[jj] = p
                    p_delegate = MyDelegate(addr)
                    delegate_global[jj] = p_delegate
                    p.withDelegate(p_delegate)
                    arduino_time[jj] = time.time()
                    post_time[jj] = time.time()
                    get_time[jj] = time.time()
                    print("Connected to "+addr+" at index: "+str(jj))                    
                    perif_loop(p,jj)                                       
        except:
            print("failed to connect to "+addr)
            continue

addr_var = get_addresses()

delegate_global = []
perif_global = []
arduino_time = [] 
post_time = []
get_time = []
plant_index_time = []
state_pump = []
NDVI = dict()
air_temp_avg = []
air_hum_avg = []
soil_hum_avg = []
water_level_avg = []


NDVI = init_NDVI(NDVI,addr_var)
[delegate_global.append(0) for ii in range(len(addr_var))]
[perif_global.append(0) for ii in range(len(addr_var))]
[arduino_time.append(time.time()) for ii in range(len(addr_var))]
[post_time.append(time.time()) for ii in range(len(addr_var))]
[get_time.append(time.time()) for ii in range(len(addr_var))]
[plant_index_time.append(time.time()) for ii in range(len(addr_var))]
[state_pump.append('OFF') for ii in range(len(addr_var))]
[air_temp_avg.append(0) for ii in range(len(addr_var))]
[air_hum_avg.append(0) for ii in range(len(addr_var))]
[soil_hum_avg.append(0) for ii in range(len(addr_var))]
[water_level_avg.append(0) for ii in range(len(addr_var))]

ex = futures.ProcessPoolExecutor(max_workers = os.cpu_count())
results = ex.map(establish_connection,addr_var)
