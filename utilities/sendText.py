from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = 'AC2844079cf7bdaf12803638a1c186e3a4'
# Your Auth Token from twilio.com/console
auth_token  = 'b385c3cec045cb08807004a994e818cc'

client = Client(account_sid, auth_token)

def sendText(body):
    message = client.messages.create(
    to="+18166748367", 
    from_="+18162031209",
    body=body)