#include <SoftwareSerial.h> //serial communication for bluetooth
#include <Adafruit_Sensor.h>
#include "DHT.h"
#include "crc8.h"

#define DHTPIN 7     // what pin we're connected to
#define DHTTYPE DHT22   // DHT 22  (AM2302)
#define SOILPIN A0  // Soil Sensor Pin
#define trigPin 5
#define echoPin 6
#define RELAY_PIN 8

#define TIME_BETWEEN_SENSORS_MEASUREMENTS_1S  1000
#define TIME_BETWEEN_SENSORS_MEASUREMENTS_30S 30000

const int AirValue   = 1000;//567;  
const int WaterValue = 650;//273; 

SoftwareSerial ble (3,2);
DHT dht(DHTPIN, DHTTYPE);

char ble_data_send[100];
char ble_data_received;
byte msg_received[100];
String cmd;
float soilmoisturepercent,air_humidity,air_temperature;
int state_sensors, state_pump, state_rx, distance, soilMoistureValue, cnt_pump_cycles, reading_sensors_time;
unsigned long old_time_sensors, old_time_ultrasound, old_time_pump,display_time;

void setup() 
{
  // Communication Setup
  ble.begin(9600);
  Serial.begin(9600); 
  dht.begin();
  // IO Setup
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);
  // Init Global Variables    
  cmd = "";
  soilMoistureValue = 0; 
  soilmoisturepercent = 0;
  distance = 0;
  air_humidity = 0;
  air_temperature = 0;
  cnt_pump_cycles = 0;
  //Init States
  state_sensors = 100;
  state_pump = 30;
  state_rx = 0;
  reading_sensors_time = TIME_BETWEEN_SENSORS_MEASUREMENTS_30S;
  //Init Timers
  old_time_sensors = millis();
  old_time_pump = millis(); 
  old_time_ultrasound = micros(); 
}

void loop() 
{  
  //Check Sensors every reading_sensors_time
  if((millis() - old_time_sensors) > reading_sensors_time && state_sensors == 100)
  {
    state_sensors = 0;
    
  }

  // Ultrasound Reading
  if (state_sensors == 0)
  {
    //Pulse Low
    digitalWrite(trigPin, LOW);
    state_sensors = 1;
    old_time_ultrasound = micros();
  }
  else if((state_sensors == 1) && (micros() - old_time_ultrasound > 4))
  {
    //Pulse High
    digitalWrite(trigPin, HIGH);
    state_sensors = 2;  
    old_time_ultrasound = micros();
  }
  else if((state_sensors == 2) && (micros() - old_time_ultrasound > 10))
  {    
    //Pulse Low
    digitalWrite(trigPin, LOW);
    // Read the echoPin, pulseIn() returns the duration (length of the pulse) in microseconds:
    long duration = pulseIn(echoPin, HIGH);
        
    // Calculate speed of sound in m/s:
    float speedofsound = 331.3+(0.606*air_temperature);
    // Calculate the distance in cm:
    distance = duration*(speedofsound/10000)/2; 
    state_sensors = 50;      
  }

  // Other Sensors Readings
  if (state_sensors == 50)
  { 
    // Read humidity as %  
    air_humidity = dht.readHumidity();
    // Read temperature as Celsius
    air_temperature = dht.readTemperature();

    for (int i = 0; i < 16; i++)
    {
      //Read Soil Sensor
      soilMoistureValue = analogRead(SOILPIN);      
      soilmoisturepercent += (float)(map(soilMoistureValue, AirValue, WaterValue, 0, 10000)/100.0);
    }
    soilmoisturepercent /= 16;

    //Serial.println(soilmoisturepercent);
    
    if(soilmoisturepercent < 0)
    {
      soilmoisturepercent = 0;
    }
    else if(soilmoisturepercent > 100)
    {
      soilmoisturepercent = 100;
    }
    
    // Check if any reads failed and exit early
    if (isnan(air_humidity) || isnan(air_temperature)) 
    {
      Serial.println("Failed to read from DHT sensor!");
      state_sensors = 100;
    }
    else
    {      
      byte data[15];
      data[0] = (byte)0xaf;
      data[1] = (byte)0x01;
      int temp_int = (int)air_temperature;
      int temp_dec = air_temperature*100-temp_int*100;
      data[2] = (byte)temp_int;
      data[3] = (byte)temp_dec;

      int hum_int = (int)air_humidity;
      int hum_dec = air_humidity*100-hum_int*100;
      data[4] = (byte)hum_int;
      data[5] = (byte)hum_dec;

      data[6] = (byte)0x01;
      int soilmoisturepercent_int = (int)soilmoisturepercent;
      int soilmoisturepercent_dec = soilmoisturepercent*100 - soilmoisturepercent_int*100;
      data[7] = (byte)soilmoisturepercent_int;
      data[8] = (byte)soilmoisturepercent_dec;

      data[9] = (byte)0x01;
      int distance_int = (int)distance;
      int distance_dec = distance*100 - distance_int*100;
      data[10] = (byte)distance_int;
      data[11] = (byte)distance_dec;
      data[12] = (byte)0xdd;

      int crc = 0;
      crc = CRC8(data, 13);
      data[13] = (byte)crc;
      
      // TX
      for (int i=0; i < 14; i++)
      {
        ble.write(data[i]);
      }

      /*Serial.print("Humidity: "); 
      Serial.print(air_humidity);
      Serial.print(" %\t");
      Serial.print("Temperature: "); 
      Serial.print(air_temperature);      
      Serial.println(" *C\t");
      Serial.print("Soil: "); 
      Serial.print(soilmoisturepercent);
      Serial.println(" %\t");
      Serial.print("Water: "); 
      Serial.print(distance);
      Serial.println(" cm\n");*/

     state_sensors = 100;
    }
    old_time_sensors = millis();
  }
  
  // RX
  ble_data_received = ble.read(); 
  
  if (int(ble_data_received)!=-1 and int(ble_data_received)!=42)
  {         
      cmd+=char(ble_data_received);
     
      if(cmd.substring(9) == "ON" && state_rx == 0)
      {        
        state_rx = 1;        
      }
      else if(state_rx == 1)
      {     
        for(int i = 0; i < 12; i++)
        {
          msg_received[i] = byte(cmd[i]);
          
        }
        int crc = 0;
        crc = CRC8(msg_received,12);
        state_rx = 0;
        if (crc == 0)
        {          
          if (state_pump == 30)
          {
            state_pump = 0;
            reading_sensors_time = TIME_BETWEEN_SENSORS_MEASUREMENTS_1S;
            Serial.println("ON");
          }                         
        }
        cmd = "";        
      }
      else if(cmd.substring(9) == "OFF" && state_rx == 0)
      {                      
        state_rx = 2;        
      } 
      else if(state_rx == 2)
      {       
        for(int i = 0; i < 13; i++)
        {
          msg_received[i] = byte(cmd[i]);          
        }
        int crc = 0;
        crc = CRC8(msg_received,13);
        state_rx = 0;
        if (crc == 0)
        {
          state_pump = 20; 
          reading_sensors_time = TIME_BETWEEN_SENSORS_MEASUREMENTS_30S;
          Serial.println("OFF");        
        }
        cmd = "";
      }
   }

   //Pump cycle
   if(state_pump == 0)
   {     
     // Emergency Condition
     if (cnt_pump_cycles > 20)
     {
       state_pump = 20;
       cnt_pump_cycles = 0;
     }
     else
     {
       digitalWrite(RELAY_PIN, LOW);
       state_pump = 1;
       old_time_pump = millis();
     }     
   }
   else if ((state_pump == 1) && ((millis() - old_time_pump) >= 200))
   {      
      state_pump = 2;
      old_time_pump = millis();      
   }
   else if((state_pump == 2))
   {
      digitalWrite(RELAY_PIN, HIGH);
      state_pump = 3;
      old_time_pump = millis();      
   }
   else if((state_pump == 3) && ((millis() - old_time_pump) >= 10000))
   {    
      state_pump = 0;
      old_time_pump = millis();   
      cnt_pump_cycles++;   
   }
   else if(state_pump == 20)
   {
      digitalWrite(RELAY_PIN, HIGH);
      old_time_pump = millis();
      state_pump = 30;   
      reading_sensors_time = TIME_BETWEEN_SENSORS_MEASUREMENTS_1S;   
   }
}
