import pandas as pd
import codecs
import requests

fname = 'cities.json'

section_download = False
if section_download:
    url = r'https://raw.githubusercontent.com/lutangar/cities.json/master/cities.json'
    r = requests.get(url, allow_redirects=True)
    open(fname, 'wb').write(r.content)
df = pd.read_json(codecs.open(fname, 'r', 'utf-8'))
ildf = df[df['country'].eq("IL")].sort_values(by='name')
