import time
import datetime
from AI import NDVI_Calc
from sharing import storeData

old_time = time.time()
set_hour = True

while(1):
	if time.time() - old_time > 30:								
		old_time = time.time()

		now = datetime.datetime.now()
		my_time_string = "11:33:00"
		my_datetime = datetime.datetime.strptime(my_time_string, "%H:%M:%S")			
		my_datetime = now.replace(hour=my_datetime.time().hour, minute=my_datetime.time().minute, second=my_datetime.time().second, microsecond=0)

		if (now > my_datetime) and (set_hour == True):
			print("Calc NVDI")
			items = NDVI_Calc()
			storeData(items)
			set_hour = False
		elif now < my_datetime:
			set_hour = True

