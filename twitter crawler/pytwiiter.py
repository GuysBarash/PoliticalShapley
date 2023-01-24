from pytwitter import Api

API_KEY = 'CmyUroSSSbcQuTxwtrkfGurSd'
API_SECRET = 'AFG6bsHVJPxU6qeNwL1WVDjyi1mbCfZ6lSFA6iBl2ZYO92wzfg'
BEARER_TOEKN = 'AAAAAAAAAAAAAAAAAAAAAIe7lAEAAAAAMuzmhSDWD1iq6C1AKaaFkG2gwq0%3DUR9ieE9QIhgc1MZG6LOel85yyGuPhljlsiwOpeVrMqhtdpNMM3'
ACCESS_TOKEN = '1535941834870079488-xyXcfNCmAJpa1bU0r287uNr1CDs1u5'
ACCESS_TOKEN_SECRET = 'JPpWwBsYUD020tHNxl2nGWO75p5SShkVBhWEyIfeYU5FA'

# NEW API
CLIENT_ID = 'TWJ1alJ2ajZ6aXlESkxRZDhDQ0M6MTpjaQ'
CLIENT_SECRET = 'DOmye_PDs9KXaY782alrBeuF1fSZERheK2NbCoJfmFlrOpRXAn'

# Path: twitter crawler\official_api.py
api = api = Api(client_id=CLIENT_ID, oauth_flow=True)
url, code_verifier, _ = api.get_oauth2_authorize_url()
access_token = api.generate_oauth2_access_token("https://localhost/?state=state&code=code", code_verifier)

r = api.get_user(username="Twitter")
j = 3
