API_KEY = 'CmyUroSSSbcQuTxwtrkfGurSd'
API_SECRET = 'AFG6bsHVJPxU6qeNwL1WVDjyi1mbCfZ6lSFA6iBl2ZYO92wzfg'
BEARER_TOEKN = 'AAAAAAAAAAAAAAAAAAAAAIe7lAEAAAAAMuzmhSDWD1iq6C1AKaaFkG2gwq0%3DUR9ieE9QIhgc1MZG6LOel85yyGuPhljlsiwOpeVrMqhtdpNMM3'
ACCESS_TOKEN = '1535941834870079488-xyXcfNCmAJpa1bU0r287uNr1CDs1u5'
ACCESS_TOKEN_SECRET = 'JPpWwBsYUD020tHNxl2nGWO75p5SShkVBhWEyIfeYU5FA'

# NEW API
CLIENT_ID = 'TWJ1alJ2ajZ6aXlESkxRZDhDQ0M6MTpjaQ'
CLIENT_SECRET = 'DOmye_PDs9KXaY782alrBeuF1fSZERheK2NbCoJfmFlrOpRXAn'

# Path: twitter crawler\official_api.py
import tweepy
import json
import pandas as pd
import os
import sys
import shutil
from pytwitter import Api
import requests

if __name__ == '__main__':
    # Set up your API credentials

    # Set up the request headers
    headers = {
        "Authorization": BEARER_TOEKN
    }

    # Make the GET request
    response = requests.get("https://api.twitter.com/2/tweets/search/recent", headers=headers)

    # Print the response
    print(response.json())

    j = 3
