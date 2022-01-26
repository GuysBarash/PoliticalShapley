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


class Wikidata:
    def __init__(self):
        self.hashed = dict()
        self.dataclient = Client()

    def get(self, q):
        if q in self.hashed:
            return self.hashed[q]
        else:
            entity = self.dataclient.get(q, load=True)
            label = str(entity.label)
            self.hashed[q] = label
            return label

    def get_entity(self, q):
        entity = self.dataclient.get(q, load=True)
        return entity

    def get_property(self, entity, p):
        pq = entity.attributes['claims'][p.upper()][0]['mainsnak']['datavalue']['value']['id']
        l = self.get(pq)
        return l


def clear_folder(path, delete_if_exist=True):
    if os.path.exists(path) and delete_if_exist:
        all_items_to_remove = [os.path.join(path, f) for f in os.listdir(path)]
        for item_to_remove in all_items_to_remove:
            if os.path.exists(item_to_remove) and not os.path.isdir(item_to_remove):
                os.remove(item_to_remove)
            else:
                shutil.rmtree(item_to_remove)

    if not os.path.exists(path):
        os.makedirs(path)


class KnessetCrawler:
    def __init__(self, restart=False):
        self.knesset_df = pd.DataFrame(columns=['knesset_number', 'start_year', 'end_year', 'knesset_wiki'])
        self.knesset_sum_df = None
        self.mk_df = None

        self.wikilinks_base = 'https://en.wikipedia.org'
        self.list_of_lists_url = r'https://en.wikipedia.org/wiki/Lists_of_Knesset_members'
        self.output_dir = os.path.join(os.path.dirname(__file__), 'CrawlerScans')
        clear_folder(self.output_dir, delete_if_exist=restart)

        self.all_knesset_df_path = os.path.join(self.output_dir, 'summed_knesset_df.csv')
        self.knesset_df_path = os.path.join(self.output_dir, 'knessets_df.csv')
        self.per_knesset_df_path = os.path.join(self.output_dir, 'knesset_@_df.csv')
        self.mk_df_path = os.path.join(self.output_dir, 'mk_df.csv')

        self.knesset_dict = dict()

        self.wikidata = Wikidata()

    def get_per_knesset_df_path(self, knesset_id):
        p = self.per_knesset_df_path.replace('@', str(knesset_id))
        return p

    def fetch_knessets(self):
        if os.path.exists(self.knesset_df_path):
            print("knessets_df already exists. Loading.")
            self.knesset_df = pd.read_csv(self.knesset_df_path)
        else:
            website_url = requests.get(self.list_of_lists_url).text
            soup = BeautifulSoup(website_url, 'lxml')
            links = soup.find_all('a')
            for l in links:
                if l.has_attr('title'):
                    if "List of members of the" in l.get('title'):
                        content = l.contents[0]
                        if re.search('[0-9]', content) is not None:
                            link = self.wikilinks_base + l.get('href')

                            numbers = re.findall('[0-9]+', content)
                            if len(numbers) < 3:
                                numbers += [numbers[-1]]
                            if len(numbers[-1]) == 2:
                                decade = numbers[1][:2]
                                full_year = decade + numbers[-1]
                                numbers[-1] = full_year
                            numbers = [int(n) for n in numbers]

                            self.knesset_df.loc[numbers[0]] = numbers + [link]

            self.knesset_df.to_csv(self.knesset_df_path, encoding='utf-8-sig', index=False)
        print(f'Knessets detected: {self.knesset_df.shape[0]}')

    def extract_mks_from_all_knessets(self):
        for r_idx, r in self.knesset_df.iterrows():
            knesset_number = r['knesset_number']
            self.get_mks_from_knesset_wiki(knesset_number)

        alldf = None
        for k in self.knesset_dict.keys():
            xdf = self.knesset_dict[k]
            if alldf is None:
                alldf = xdf.copy()
            else:
                for c in alldf.columns:
                    if c not in xdf.columns:
                        xdf[c] = ''
                for c in xdf.columns:
                    if c not in alldf.columns:
                        alldf[c] = ''
                alldf = pd.concat([alldf, xdf], sort=False, ignore_index=True)

        self.knesset_sum_df = alldf
        self.knesset_sum_df.to_csv(self.all_knesset_df_path, encoding='utf-8-sig', index=False)
        print(f"All Knesset summary at: {self.all_knesset_df_path}")

        if not (os.path.exists(self.mk_df_path)):
            uid = self.knesset_sum_df['Member wikidata token'].unique()
            self.mk_df = pd.DataFrame(index=uid)
            self.mk_df['wikidata token'] = uid
            self.mk_df['Name'] = None
            self.mk_df['Wikipedia'] = None
            for uid_t in uid:
                r = self.knesset_sum_df[self.knesset_sum_df['Member wikidata token'].eq(uid_t)].iloc[0]
                self.mk_df.loc[uid_t, ['Name', 'Wikipedia']] = r[['Member', 'Member wiki']].values

            self.mk_df.to_csv(self.mk_df_path, encoding='utf-8-sig', index=True)
            print(f"MK summary at: {self.mk_df_path}")
        else:
            self.mk_df = pd.read_csv(self.mk_df_path, index_col=0)

    def get_mks_from_knesset_wiki(self, knesset_number):
        r = self.knesset_df[self.knesset_df['knesset_number'].eq(knesset_number)].iloc[0]
        knesset_url = r['knesset_wiki']
        df_path = self.get_per_knesset_df_path(knesset_number)
        if os.path.exists(df_path):
            print(f"Knesset {knesset_number} Already exists. Loading")
            df = pd.read_csv(df_path)
        else:
            # Get base df
            print(f"Reading from: {knesset_url}")
            dfs = pd.read_html(knesset_url, header=0)
            df = dfs[0]
            if 'Part of' in df.columns[0]:
                df = dfs[1]

            replace_cols = dict()
            replace_cols['Member'] = ['MK', 'Name']
            replace_cols['Note'] = ['Notes']
            replace_cols['Party'] = ['Party.1']
            drop_cols = ['Alliance', 'Faction']
            for drop_col in drop_cols:
                if drop_col in df.columns:
                    df = df.drop(drop_col, axis=1)

            for right_col in replace_cols.keys():
                bad_cols = replace_cols[right_col]
                for bad_col in bad_cols:
                    if bad_col in df.columns:
                        df[right_col] = df[bad_col]
                        df = df.drop(bad_col, axis=1)

            special_names = dict()
            special_names['Party'] = ['Alliance.1', 'Faction.1']
            for special_name_key in special_names.keys():
                for special_name_src in special_names[special_name_key]:
                    if special_name_src in df.columns:
                        df[special_name_key] = df[special_name_src]
                        df = df.drop(special_name_src, axis=1)

            df['Member'] = df['Member'].str.replace('|', '')

            if 'Note' not in df.columns:
                pattern = '(.*)[\s]*\((.*)\)'
                noted_idx = df['Member'].str.contains('\(')
                df['Note'] = ''
                q = df.loc[noted_idx, 'Member'].str.extract(pattern, expand=True)
                df.loc[noted_idx, 'Note'] = q[1]
                df.loc[noted_idx, 'Member'] = q[0]
            else:
                df['Note'] = df['Note'].fillna('')

            seats_detected = False
            if 'Seats' not in df.columns:
                pattern = '(.*)[\s]*\(([0-9]+)\)'
                noted_idx = df['Party'].str.contains('\(')
                df['Seats'] = ''
                q = df.loc[noted_idx, 'Party'].str.extract(pattern, expand=True)
                df.loc[noted_idx, 'Seats'] = q[1]
                df.loc[noted_idx, 'Party'] = q[0]
                # if noted_idx.sum() > 0:
                #     seats_detected = True

            df['Party'] = df['Party'].str.replace(r'[ \t]+$', '')
            df['Party'] = df['Party'].str.replace(r'\[.*\]$', '')
            df['Party'] = df['Party'].str.replace(r'[ \t]+$', '')
            df['Member'] = df['Member'].str.replace(r'\[.*\]$', '')
            df['Member'] = df['Member'].str.replace(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]', '')
            df['Member'] = df['Member'].str.replace(r'[ \t]+$', '')

            # Find tables
            website_url = requests.get(knesset_url).text
            soup = BeautifulSoup(website_url, 'lxml')
            My_table = soup.find('ul')
            rows = My_table.find_all('li')
            segs = dict()
            for row in rows:
                m = re.search('([0-9]+)', row.text)
                if m is None:
                    continue
                seats = int(m.group(0))
                l = row.find('a')
                latt = l.attrs

                wikilinks_base = 'https://en.wikipedia.org'
                wikilink = wikilinks_base + latt['href']

                party_name = l.text
                segs[party_name] = [seats, wikilink]

            df['Party wiki'] = ''
            for party_k in segs.keys():
                p_seats = segs[party_k][0]
                p_link = segs[party_k][1]
                idx = df['Party'].eq(party_k)
                if idx.sum() == 0:
                    special_cases = dict()
                    special_cases['Flatto-Sharon'] = 'Development and Peace'
                    special_cases['Arab Democratic Party'] = "Mada-Ra'am"

                    idx = df['Party'].str.contains(special_cases.get(party_k, party_k))
                    if idx.sum() == 0:
                        print(f"Unknown: {party_k}")
                df.loc[idx, 'Seats'] = p_seats
                df.loc[idx, 'Party wiki'] = p_link

            # get raw links
            website_url = requests.get(knesset_url).text
            soup = BeautifulSoup(website_url, 'lxml')
            My_table = soup.find('table')
            links = soup.findAll('a')
            sr_links = pd.Series()
            for l in links:
                if len(l.contents) > 0:
                    superscript = '[⁰¹²³⁴⁵⁶⁷⁸⁹]'
                    name = l.contents[0]
                    name = re.sub(superscript, '', str(name))

                    t_link = l.get('href')
                    if t_link is not None and re.search(r'/wiki/', t_link) is not None:
                        wikilinks_base = 'https://en.wikipedia.org'
                        sr_links[name] = wikilinks_base + t_link
                    else:
                        j = 3
            idx_to_links_sr = pd.Series(index=df.index)

            df['Member wiki'] = ''
            prty_q_dict = dict()
            for r_idx, r in tqdm(df.iterrows(), total=df.shape[0], desc='Searching wikidata'):
                name = r['Member']
                prty = r['Party']
                # superscript = '[⁰¹²³⁴⁵⁶⁷⁸⁹]'
                # name = re.sub(superscript, '', str(name))
                # side_note = r'\[.*\]$'
                # name = re.sub(side_note, '', str(name))
                mamber_link = sr_links[name]
                member_link_title = re.search('/([^/]+)$', mamber_link).group(1)
                member_q = query_wikidata(member_link_title, mamber_link)

                prty_link = sr_links.get(prty, '')
                party_q = None
                if prty_link != '':
                    party_link_title = re.search('/([^/]+)$', prty_link).group(1)
                    if party_link_title in prty_q_dict:
                        party_q = prty_q_dict[party_link_title]
                    else:
                        party_q = query_wikidata(party_link_title, prty_link)
                        prty_q_dict[party_link_title] = party_q

                df.loc[r_idx, 'Member wiki'] = mamber_link
                df.loc[r_idx, 'Member wikidata token'] = member_q
                df.loc[r_idx, 'Party wiki'] = prty_link
                df.loc[r_idx, 'Party wikidata token'] = party_q

            print(f"Knesset {knesset_number} Extracted and saved")
            df['Knesset_id'] = knesset_number
            if knesset_number == 17:
                seats = dict()
                seats["Kadima"] = 29
                seats["Labor-Meimad"] = 19
                seats["Shas"] = 12
                seats["Gil"] = 7
                seats["Likud"] = 12
                seats["Yisrael Beiteinu"] = 11
                seats["National Union-NRP"] = 9
                seats["United Torah Judaism"] = 6
                seats["Meretz-Yachad"] = 5
                seats["United Arab List"] = 4
                seats["Hadash"] = 3
                seats["Balad"] = 3
                df['Seats'] = df['Party'].map(seats)

            df.to_csv(df_path, encoding='utf-8-sig', index=False)

        self.knesset_dict[knesset_number] = df

    def extract_mks_from_wikidata(self):
        qidf = pd.DataFrame()
        qidf_path = os.path.join(self.output_dir, 'qidf.csv')
        if os.path.exists(qidf_path):
            qidf = pd.read_csv(qidf_path, index_col=0)
        for idx, (r_idx, r) in enumerate(
                tqdm(self.mk_df.iterrows(), total=self.mk_df.shape[0], desc='Wikidata queries')):
            qid = r_idx
            if qid in qidf.index:
                continue
            try:
                sr = self.get_mks_wikidata(qid)
                for c in sr.index:
                    if c not in qidf.columns:
                        qidf[c] = None
                qidf.loc[qid, sr.index] = sr
            except Exception as e:
                print(f"Failed in: {qid}")

            if idx % 50 == 0 and idx > 0:
                qidf.to_csv(qidf_path, encoding='utf-8-sig')
        qidf.to_csv(qidf_path, encoding='utf-8-sig')

        countPerc = qidf.isna().mean()
        bar = 0.5
        countPerc = countPerc[countPerc < bar]
        qidf = qidf[countPerc.index]
        self.mk_df[qidf.columns] = qidf
        self.mk_df.to_csv(self.mk_df_path, encoding='utf-8-sig')

    def get_mks_wikidata(self, qid, spin=10):
        import requests

        url = 'https://query.wikidata.org/sparql'
        query = '''
        SELECT ?wdLabel ?ps_Label ?wdpqLabel ?pq_Label {
          VALUES (?company) {(wd:@@@qid@@@)}
        
          ?company ?p ?statement .
          ?statement ?ps ?ps_ .
        
          ?wd wikibase:claim ?p.
          ?wd wikibase:statementProperty ?ps.
        
          OPTIONAL {
          ?statement ?pq ?pq_ .
          ?wdpq wikibase:qualifier ?pq .
          }
        
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        } ORDER BY ?wd ?statement ?ps_
        '''.replace('@@@qid@@@', str(qid))
        g_time = datetime.datetime.now()
        for i in range(spin):
            l_time = datetime.datetime.now()
            v = False
            try:
                r = requests.get(url, params={'format': 'json', 'query': query})
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
        pdict = {mt['wdLabel']['value']: mt['ps_Label']['value'] for mt in m}
        psr = pd.Series(pdict)
        return psr

    def fix_place_of_birth(self):
        fname = 'cities.json'
        pobdf = pd.read_json(codecs.open(fname, 'r', 'utf-8'))
        pobdf = pobdf.set_index('name').sort_values(by=['country', 'name'])
        l = self.mk_df['place of birth'].unique().tolist()
        missed = list()
        self.mk_df['Country of birth'] = np.nan
        for cl in l:
            try:
                r = pobdf.loc[cl]
                if type(r['country']) is str:
                    self.mk_df.loc[self.mk_df['place of birth'].eq(cl), 'Country of birth'] = r['country']
                else:
                    print(f"Strange city: {cl}")
            except KeyError as e:
                print(f"Skipping: {cl}")
                missed += [cl]
        j = 3


def query_wikidata(title, wikipage=None):
    base = r'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles=@&format=json'
    q_url = base.replace('@', title)

    import urllib.request, json

    with urllib.request.urlopen(q_url) as url:
        data = json.loads(url.read().decode())

    mq = re.search(r'(Q[0-9]+)', str(data))
    if mq is not None:
        q = mq.group(1)
    elif wikipage is not None:
        try:
            website_url = requests.get(wikipage).text
        except Exception as e:
            j = 3
        soup = BeautifulSoup(website_url, 'lxml')
        links = soup.find_all('a')
        links = [lt.attrs['href'] for lt in links if
                 ('href' in lt.attrs) and ('https://www.wikidata.org/wiki/' in lt.attrs['href'])]
        mq = re.search(r'(Q[0-9]+)', str(links[0]))
        if mq is not None:
            q = mq.group(1)
        else:
            q = None
            print(f"No wikidata for: {title}")
    else:
        q = None
        print(f"No wikidata for: {title}")
    return q


if __name__ == '__main__':
    q = '''SELECT ?townLabel ?countryLabel ?country_population ?town ?country
WHERE
{
  VALUES ?town_or_city {
    wd:Q3957
    wd:Q515
  }
  ?town   wdt:P31/wdt:P279* ?town_or_city.
  ?country wdt:P31/wdt:P279* wd:Q3624078.
  
  
  ?town  wdt:P17 ?country.
  ?country  wdt:P36 ?town. # Capital of
  ?country wdt:P1082 ?country_population.
  ?town wdt:P1082 ?city_population.
           
  FILTER( ?country_population >= "100000"^^xsd:integer )
  FILTER( ?city_population >= "10000"^^xsd:integer )

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY DESC(?country_population)'''
    wikidata_sql()

if __name__ == '__main__':
    crawler = KnessetCrawler(False)
    crawler.fetch_knessets()
    crawler.extract_mks_from_all_knessets()
    crawler.extract_mks_from_wikidata()
    crawler.fix_place_of_birth()

    mk_df, knesset_df, mkpk_df = crawler.mk_df, crawler.knesset_df, crawler.knesset_sum_df

    # Gender
    gender = pd.Series(index=mk_df['Name'], data=mk_df['sex or gender'].values)
    mkpk_df['Gender'] = mkpk_df['Member'].replace(gender)
    mkpk_df['Female'] = mkpk_df['Gender'].eq('female').astype(int)
    g = mkpk_df.groupby(by='Knesset_id')['Female'].mean()

    # Longest serving
    services = mkpk_df.groupby('Member')['Seats'].count().sort_values(ascending=False)
    services = services[services > 7]

    # Most parties
    zigzag = mkpk_df.groupby('Member')['Party'].nunique().sort_values(ascending=False)
    zigzag = zigzag[zigzag > 1]

    # Knesset duration
    klong = pd.Series(index=knesset_df['knesset_number'],
                      data=(knesset_df['end_year'] - knesset_df['start_year']).values)
    klong = klong.iloc[:-1]  # Last one is still running
