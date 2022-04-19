import datetime
import os
import shutil
import re
import time
import random
import json
from copy import deepcopy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import queue

from tqdm import tqdm

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import itertools
from difflib import SequenceMatcher

if __name__ == '__main__':
    section_musks = True
    if section_musks:
        path = r"C:\school\PoliticalShapley\statistics\data\lms\mosques_raw.csv"
        root_path = os.path.dirname(path)
        mosque_path = os.path.join(root_path, f'Mosques.csv')
        msqdf = pd.read_csv(path, encoding='utf-8-sig')
        msqdf = msqdf[['town', 'mosque']]

        towncode_path = r"C:\school\PoliticalShapley\statistics\Results\votes distributions.csv"
        towndf = pd.read_csv(towncode_path, encoding='utf-8-sig')
        towndf = towndf[['town', 'uid']].sort_values(by='town', ascending=False)

        # Replace
        for from_name, to_name in [('יפה', 'תל אביב יפו'),
                                   ('באקה', 'באקה אלגרביה'),
                                   ('סאלם', 'סולם'),
                                   ('סולם  ', 'סולם'),
                                   ('גת', 'גית'),
                                   ('כפר מסר', 'כפר מצר'),
                                   ('גוש חלב', 'גש גוש חלב'),
                                   ('בועינה', 'בענה'),
                                   ('אלמגאר', 'מגאר'),
                                   ('פרדיס', 'פוריידיס'),
                                   ('גדידה', 'גדיידהמהכר'),
                                   ('תרשיחא', 'מעלותתרשיחא'),
                                   ('סידנא עלי', 'הרצליה'),
                                   ("חג'אג'רה", 'כעביהטבאשחגאגרה'),
                                   ('בועינה', 'בענה'),

                                   ]:
            idxs = msqdf['town'].eq(from_name)
            msqdf.loc[idxs, 'town'] = to_name


        # Find similarities
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()


        col = 'town'
        msqdf['town suggest'] = None
        msqdf['town val'] = None
        msqdf['uid'] = None
        for idx in msqdf[col].index:
            a = msqdf.loc[idx, col]
            if pd.isna(a):
                continue
            bs = towndf['town'].values.tolist()
            uids = towndf['uid'].values.tolist()
            bv = [similar(a, b) for b in bs]
            opt_idx = np.argmax(bv)
            msqdf.loc[idx, ['town suggest', 'town val', 'uid']] = bs[opt_idx], bv[opt_idx], uids[opt_idx]

        msqdf['match'] = msqdf['town val'] > 0.86

        xdf = msqdf.loc[msqdf['match']]
        df = pd.DataFrame(columns=['town', 'mosque', 'real town', 'uid'], index=xdf.index)
        df.loc[xdf.index] = xdf[['town', 'mosque', 'town suggest', 'uid']].to_numpy()

        # msqdf = msqdf[~msqdf['match']]
        msqdf = msqdf.sort_values(by='town val', ascending=False)
        g = df.groupby(by='town')
        odf = pd.DataFrame(index=df['town'].unique(), columns=['uid', 'Mosques'])
        odf['Mosques'] = g['uid'].count()
        odf['uid'] = g['uid'].agg(pd.Series.mode)
        odf.to_csv(mosque_path, encoding='utf-8-sig')

    section_police = False
    if section_police:
        path = "C:\school\PoliticalShapley\statistics\data\lms\crime police.csv"
        root_path = os.path.dirname(path)
        police_path = os.path.join(root_path, f'police_info.csv')

        section_clean_police = True
        if section_clean_police:
            df = pd.read_csv(path, encoding='utf-8-sig')
            df = df.loc[:, ~df.isna().all()]
            remove_towns = ['-', 'Total', 'אחר']
            df = df[~df['יישוב'].isin(remove_towns)]

            cols = ['יישוב',
                    'תיאור עבירה',
                    'סה"כ',
                    ]
            df = df[cols]

            type_of_crime = [t for t in df['תיאור עבירה'].unique().tolist() if (~pd.isna(t))][1:]
            cols = ['town'] + type_of_crime
            towns = [t for t in df['יישוב'].unique().tolist() if t != '-' and ~pd.isna(t) and t != 'Total']
            idx = towns
            xdf = pd.DataFrame(columns=cols, index=idx)
            xdf['town'] = xdf.index.to_list()

            for c in type_of_crime:
                m = df[df['תיאור עבירה'].eq(c)]
                m = m.set_index('יישוב')
                m = m['סה"כ']
                xdf.loc[m.index, c] = m.values
            xdf = xdf.drop('town', axis=1)

            for c in xdf.columns:
                xdf[c] = xdf[c].fillna('0').str.replace(',', '').astype(int)
            xdf = xdf.drop('-', axis=1)

            xdf.to_csv(police_path, encoding='utf-8-sig')

        section_add_code = True
        if section_add_code:
            policedf = pd.read_csv(police_path, encoding='utf-8-sig', index_col=0)
            towncode_path = r"C:\school\PoliticalShapley\statistics\Results\votes distributions.csv"
            towndf = pd.read_csv(towncode_path, encoding='utf-8-sig')
            towndf = towndf[['town', 'uid']]

            policedf['town'] = policedf.index.values
            policedf['town'] = policedf['town'].replace('מועצה אזורית ', '')

            police_set = set(policedf['town'])
            print(f"Original police towns: {len(police_set)}")
            vote_set = set(towndf['town'])

            common = police_set.intersection(vote_set)
            only_in_police = police_set.difference(vote_set)
            only_in_vote = vote_set.difference(police_set)
            print(f"Already solved: {len(common)}\tTo solve: {len(only_in_police)}")

            r = pd.DataFrame(index=range(max(len(only_in_police), len(only_in_vote))))
            r.loc[range(len(only_in_police)), 'police'] = sorted(list(only_in_police))
            r.loc[range(len(only_in_vote)), 'vote'] = sorted(list(only_in_vote))


            def similar(a, b):
                return SequenceMatcher(None, a, b).ratio()


            r['Suggest'] = None
            r['val'] = None
            for idx in r['police'].index:
                a = r.loc[idx, 'police']
                if pd.isna(a):
                    continue
                bs = r['vote'].values.tolist()
                bv = [similar(a, b) for b in bs]
                opt_idx = np.argmax(bv)
                r.loc[idx, ['Suggest', 'val']] = bs[opt_idx], bv[opt_idx]
            r = r[['police', 'Suggest', 'val']]
            r = r.sort_values(by=['val'], ascending=False)
            r = r[~r['police'].isna()]

            r = r[r['police'].isin(["ע'ג'ר", 'ברנר']) | (r['val'] >= 0.8)]
            r = r[~r['police'].isin(["שער הנגב", "תמר"])]
            convertion = {r['police']: r['Suggest'] for _, r in r.iterrows()}

            for k, v in convertion.items():
                idxs = policedf['town'].eq(k)
                policedf.loc[idxs, 'town'] = v

            policedf = policedf.merge(towndf, left_on='town', right_on='town', how='outer')
            policedf = policedf[~policedf['עבירות בטחון'].isna()]
            policedf = policedf[~policedf['uid'].isna()]

            common = police_set.intersection(vote_set)
            only_in_police = police_set.difference(vote_set)
            print(f"solved after adaption: {policedf['town'].unique().shape[0]}")
            policedf.to_csv(police_path, encoding='utf-8-sig')

if __name__ == '__main__':
    print("END OF CODE.")
