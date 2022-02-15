import datetime
import os
import shutil
import re
import time

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from IPython.display import display, HTML

import codecs
from wikidata.client import Client

from tqdm import tqdm


def clear_folder(path, clear_is_exists=True):
    if clear_is_exists and os.path.exists(path):
        all_items_to_remove = [os.path.join(path, f) for f in os.listdir(path)]
        for item_to_remove in all_items_to_remove:
            if os.path.exists(item_to_remove) and not os.path.isdir(item_to_remove):
                os.remove(item_to_remove)
            else:
                shutil.rmtree(item_to_remove)

    if not os.path.exists(path):
        os.makedirs(path, mode=0o777)

    time.sleep(0.1)


def wikidata_sql(q=None, spin=5):
    if q is None:
        q = r'''
    SELECT ?townLabel ?countryLabel ?country_population ?town ?country
    WHERE
    {
      VALUES ?town_or_city {
        wd:Q3957
        wd:Q515
      }
      ?town   wdt:P31/wdt:P279* ?town_or_city.
      ?country wdt:P31/wdt:P279* wd:Q3624078.


      ?town  wdt:P17 ?country.
      # ?country  wdt:P36 ?town. # Capital of
      ?country wdt:P1082 ?country_population.
      ?town wdt:P1082 ?city_population.

      FILTER( ?country_population >= "100000"^^xsd:integer )
      FILTER( ?city_population >= "10000"^^xsd:integer )

      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ORDER BY DESC(?country_population)
        '''

    import requests

    url = 'https://query.wikidata.org/sparql'
    g_time = datetime.datetime.now()
    for i in range(spin):
        v = False
        try:
            l_time = datetime.datetime.now()
            print(f"Attempt {i + 1} at query")
            r = requests.get(url, params={'format': 'json', 'query': q})
            data = r.json()
            v = True
        except Exception as e:
            nowtime = datetime.datetime.now()
            print(f"Failed round {i + 1}\tGlobal time: {nowtime - g_time}\tRound time: {nowtime - l_time}")
            time.sleep(0.5)
        if v:
            nowtime = datetime.datetime.now()
            print(f"Success! round {i + 1}\tGlobal time: {nowtime - g_time}\tRound time: {nowtime - l_time}")
            break

    m = data['results']['bindings']
    mdict = [{mtk: mtv['value'] for mtk, mtv in mt.items()} for mt in m]
    pdf = pd.DataFrame(mdict)
    return pdf


if __name__ == '__main__':
    q = '''
SELECT DISTINCT ?conflict ?conflictLabel ?participateLabel  ?start_date ?end_date ?deaths
WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en,[AUTO_LANGUAGE]"}
  ?war (ps:P31/(wdt:P279*)) wd:Q198.  
  ?is_state (ps:P31/(wdt:P279*)) wd:Q7275.

  ?conflict p:P31 ?war.
  ?conflict wdt:P710 ?participate.
  ?participate p:P31 ?is_state.

  ?conflict wdt:P580 ?start_date.
  optional{?conflict wdt:P580 ?end_date}
  FILTER(YEAR(?start_date) >= 1946)
  }

    '''
    rdf = wikidata_sql(q)

    section_by_conflict = True
    if section_by_conflict:
        conflicts = rdf['conflictLabel'].unique()
        df = pd.DataFrame(columns=['Conflict', 'Start date', 'End date'] + [f'Participant {i + 1}' for i in range(50)])
        for idx, conflict in tqdm(enumerate(conflicts), desc='Conflicts'):
            cdf = rdf[rdf['conflictLabel'].eq(conflict)]
            start_time, end_time = cdf['start_date'].min(), cdf['end_date'].max()
            df.loc[idx, ['Conflict', 'Start date', 'End date']] = [conflict, start_time, end_time]
            parts = cdf['participateLabel'].unique().tolist()
            df.loc[idx, [f'Participant {i + 1}' for i in range(len(parts))]] = parts

        non_empties = df.isna().mean() < 1
        df = df[non_empties[non_empties].index.to_list()]

        root_path = os.path.join(os.path.dirname(__file__), 'Dump')
        clear_folder(root_path, clear_is_exists=False)
        dfpath = os.path.join(root_path, 'Conflicts participants.csv')
        print(f"Creating file: {dfpath}")
        df.to_csv(dfpath, index=False, encoding='utf-8-sig')

    section_by_state = True
    if section_by_state:
        states = rdf['participateLabel'].unique()
        df = pd.DataFrame(columns=['State', 'Conflicts'] + [f'Conflict {i + 1}' for i in range(50)])
        for idx, state in tqdm(enumerate(states), desc='Conflicts'):
            cdf = rdf[rdf['participateLabel'].eq(state)]
            conflicts = cdf['conflictLabel'].unique().tolist()
            df.loc[idx, ['State', 'Conflicts']] = [state, len(conflicts)]
            df.loc[idx, [f'Conflict {i + 1}' for i in range(len(conflicts))]] = conflicts

        non_empties = df.isna().mean() < 1
        df = df[non_empties[non_empties].index.to_list()]

        root_path = os.path.join(os.path.dirname(__file__), 'Dump')
        clear_folder(root_path, clear_is_exists=False)
        dfpath = os.path.join(root_path, 'participants.csv')
        print(f"Creating file: {dfpath}")
        df.to_csv(dfpath, index=False, encoding='utf-8-sig')
