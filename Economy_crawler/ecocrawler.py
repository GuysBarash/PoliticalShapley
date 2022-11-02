import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import requests
import json
from datetime import datetime


# Get GDP through API
def get_gdp():
    # Get GDP data from API
    url = 'https://www.quandl.com/api/v3/datasets/ODA/PNG_NGDP.json?api_key=3p8zWxj9q3YzY4z4t4y8'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['dataset']['data'], columns=data['dataset']['column_names'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.sort_index()
    return df


df = get_gdp()
