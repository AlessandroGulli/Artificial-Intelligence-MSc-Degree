import pickle 

def storeData(NDVI):  	
	db = {} 
	for key in NDVI:
		db[key] = NDVI[key]   
	dbfile = open('DataNDVI', 'ab') 
	pickle.dump(db, dbfile)                   
	dbfile.close() 

def loadData():	
	dbfile = open('DataNDVI', 'rb')      
	db = pickle.load(dbfile) 
	out = {} 
	for keys in db: 
		out[keys] = db[keys] 
	dbfile.close()
	return out