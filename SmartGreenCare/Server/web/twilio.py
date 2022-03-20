from twilio.rest import Client 
 
account_sid = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' 
               
auth_token  = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' 
			   
client = Client(account_sid, auth_token) 
 

def send_wa_message(client, text):
    client.messages.create(from_='whatsapp:+14155238886',
                           body=text,      
                           to='whatsapp:+39340xxxxxxx' 
                           )
    return