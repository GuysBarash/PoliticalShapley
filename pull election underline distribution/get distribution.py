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
import requests
from bs4 import BeautifulSoup
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import queue

from selenium import webdriver
from selenium.webdriver.chrome.options import Options, DesiredCapabilities
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.support.ui import WebDriverWait

from tqdm import tqdm

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import networkx as nx
import plotly.figure_factory as ff

# Communities
from networkx.algorithms import community
import itertools
from sklearn.metrics.pairwise import euclidean_distances


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


class Voter:
    def __init__(self, votesdf, meatavoters=10):
        self.rootpath = os.path.dirname(__file__)
        self.outpath = os.path.join(self.rootpath, 'Results')
        self.datapath = os.path.join(self.rootpath, 'data')
        self.lms_path = os.path.join(self.datapath, 'lms')
        clear_folder(self.outpath, delete_if_exist=False)
        clear_folder(self.rootpath, delete_if_exist=False)
        clear_folder(self.lms_path, delete_if_exist=False)

        self.meatavoters = meatavoters
        self.rawdf = votesdf
        self.rawdf = self.rawdf[~self.rawdf['שם ישוב'].eq("מעטפות חיצוניות")]
        self.rawdf['uid'] = self.rawdf['סמל ישוב'].astype(int)

        self.metadata = list(set(df.columns[:7].to_list() + ['uid']))
        self.parties = [c for c in self.rawdf.columns if c not in self.metadata]
        self.parties = [c for c in self.parties if 'Unnamed' not in c]

        self.votesdf = self.rawdf[self.parties]
        self.votesdf = self.votesdf[self.votesdf.sum(axis=1) > 500]
        # self.votesdf = self.votesdf[self.votesdf.sum(axis=1) < 4000]
        names = self.rawdf.loc[self.votesdf.index, 'שם ישוב']
        uid = self.rawdf.loc[self.votesdf.index, 'uid']

        self.votes_dist_df = self.votesdf.div(self.votesdf.sum(axis=1), axis=0)
        self.votes_dist_df = np.round(self.votes_dist_df, 2)
        self.votes_dist_df = self.votes_dist_df.div(self.votes_dist_df.sum(axis=1), axis=0)

        nonempties = self.votes_dist_df.max() > 0
        nonempties = nonempties[nonempties]
        self.non_empy_parties = nonempties
        self.votes_dist_df = self.votes_dist_df[nonempties.index]
        self.votes_dist_df['town'] = names
        self.votes_dist_df['uid'] = uid
        self.votes_dist_df = self.votes_dist_df.set_index('town')

        d = self.votes_dist_df[[c for c in self.votes_dist_df if c in self.parties]].to_numpy()
        dist = euclidean_distances(d, d)
        self.distances = pd.DataFrame(index=names, columns=names, data=dist)

        self.final_cout = self.votesdf[self.parties].sum().sort_values(ascending=False)
        self.norm_count = self.final_cout / self.final_cout.sum()

        self.voters = None

    def add_external_data(self):
        print('<-------------------------->')
        print(self.votesdf.shape[0])
        section_ethnicity = False
        if section_ethnicity:
            ethdf = pd.read_csv(os.path.join(self.lms_path, 'ethnicity.csv'), encoding='utf-8-sig')
            ethdf['uid'] = ethdf['סמל היישוב'].str.replace(',', '').astype(int)

            edf = pd.DataFrame(index=ethdf.index)
            edf['uid'] = ethdf['uid']
            edf['Jews'] = ethdf['יהודים ואחרים (אחוזים)'].str.replace('-', '0.0').astype(float) / 100.0
            edf['Arab'] = ethdf['ערבים (אחוזים)'].str.replace('-', '0.0').replace('', '0.0').astype(float) / 100.0
            edf['Arab-Muslim'] = ethdf['מוסלמים (אחוזים מתוך האוכלוסייה הערבית)'].str.replace('-', '0.0').replace('',
                                                                                                                  '0.0').astype(
                float) / 100.0
            edf['Arab-Muslim'] *= edf['Arab']
            edf['Arab-Christian'] = ethdf['נוצרים (אחוזים מתוך האוכלוסייה הערבית)'].str.replace('-', '0.0').replace('',
                                                                                                                    '0.0').astype(
                float) / 100.0
            edf['Arab-Christian'] *= edf['Arab']
            edf['Druze'] = ethdf['דרוזים (אחוזים מתוך האוכלוסייה הערבית)'].str.replace('-', '0.0').replace('',
                                                                                                           '0.0').astype(
                float) / 100.0
            edf['Druze'] *= edf['Arab']
            edf = edf.drop('Arab', axis=1)

            self.votesdf = self.votesdf.merge(edf, left_on='uid', right_on='uid', how='outer')

        section_money = True
        if section_money:
            money = pd.read_csv(os.path.join(self.lms_path, 'paycheck.csv'), encoding='utf-8-sig')
            money['uid'] = money['סמל היישוב'].str.replace(',', '').astype(int)
            money['Salary'] = money['שכר ממוצע לחודש של שכירים (ש"ח) כלל השכירים'].str.replace(',', '').astype(float)
            money['JINI'] = money["מדד אי-השוויון שכירים (מדד ג'יני, 0 שוויון מלא)"].astype(float)
            money = money[['uid', 'Salary', 'JINI']]
            self.votesdf = self.votesdf.merge(money, left_on='uid', right_on='uid', how='outer')

        section_education = True
        if section_education:
            education = pd.read_csv(os.path.join(self.lms_path, 'education.csv'), encoding='utf-8-sig')
            education = education[~education['סמל היישוב'].isna()]
            education['uid'] = education['סמל היישוב'].str.replace(',', '').astype(int)
            education['Bagrut'] = education['אחוז זכאים לתעודת בגרות מבין תלמידי כיתות יב תשע"ט 2018/19'].str.replace(
                '-', '0.0').astype(float)
            education['Degree'] = education['השכלה גבוהה אחוז סטודנטים מתוך אוכלוסיית בני 25-20 תש"ף 2019/20']
            education = education[['uid', 'Bagrut', 'Degree']]
            self.votesdf = self.votesdf.merge(education, left_on='uid', right_on='uid', how='outer')

        section_health = True
        if section_health:
            health = pd.read_csv(os.path.join(self.lms_path, 'health.csv'), encoding='utf-8-sig')
            health = health[~health['סמל היישוב'].isna()]

            q = pd.DataFrame(index=health.index)
            q['uid'] = health['סמל היישוב'].str.replace(',', '').astype(int)
            q['under 18'] = health['teens'] / 100.0
            q['above 75'] = health['olds'] / 100.0
            q['natural growth'] = health['growth normal']
            q['density'] = health['Density'].str.replace(',', '').apply(pd.to_numeric, args=('coerce',))
            self.votesdf = self.votesdf.merge(q, left_on='uid', right_on='uid', how='outer')

        section_general = True
        if section_general:
            gendf = pd.read_csv(os.path.join(self.lms_path, 'general.csv'), encoding='utf-8-sig')
            gendf = gendf[~gendf['סמל היישוב'].isna()]

            q = pd.DataFrame(index=gendf.index)
            q['uid'] = gendf['סמל היישוב']
            q['county'] = gendf['נפה']
            q['Settelment type'] = gendf['צורת יישוב']
            q['population'] = gendf['אוכלוסייה - סך הכל'].str.replace(',', '').str.replace('-', '0').astype(float)
            q['Jews pop'] = gendf['מזה: יהודים ואחרים'].str.replace(',', '').str.replace('-', '0').astype(float) / q[
                'population']
            q['Non-Jews pop'] = gendf['מזה: ערבים'].str.replace(',', '').str.replace('-', '0').astype(float) / q[
                'population']
            self.votesdf = self.votesdf.merge(q, left_on='uid', right_on='uid', how='outer')

        self.votesdf = self.votesdf[~self.votesdf['Label'].isna()]

        section_add_means = True
        if section_add_means:
            new_cols = ['Jews', 'Arab-Muslim', 'Arab-Christian', 'Druze',
                        'Bagrut', 'Degree',
                        'under 18', 'above 75', 'natural growth', 'density',
                        'Salary', 'JINI',
                        'population', 'Jews pop', 'Non-Jews pop',
                        ]
            for c in new_cols:
                if c not in self.votesdf.columns:
                    continue
                df = self.votesdf[['Label', c]]
                g = df.groupby('Label')
                nullratio = g.agg({c: lambda x: x.isnull().mean()})
                null_max = nullratio.max().values[0]
                null_max_label = nullratio.idxmax().values[0]
                if null_max < 0.7:
                    self.voters[c] = g[c].mean()
                else:
                    print(f"Max [{c}] x [{null_max_label}] = {null_max}")

        section_add_most_common = True
        if section_add_most_common:
            new_cols = ['county', 'Settelment type'
                        ]
            for c in new_cols:
                if c not in self.votesdf.columns:
                    continue
                df = self.votesdf[['Label', c]]
                g = df.groupby('Label')
                nullratio = g.agg({c: lambda x: x.isnull().mean()})
                null_max = nullratio.max().values[0]
                null_max_label = nullratio.idxmax().values[0]
                if null_max < 0.7:
                    self.voters[c] = g[c].agg(lambda x: x.value_counts().index[0])
                else:
                    print(f"Max [{c}] x [{null_max_label}] = {null_max}")
                    print("Skipping")
        j = 3

    def cluster(self):
        # dbscan clustering
        from kneed import KneeLocator
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        prts = [c for c in self.votes_dist_df if c in self.parties]
        X = self.votes_dist_df[prts].to_numpy()

        scaler = StandardScaler()
        scaled_X = X  # scaler.fit_transform(X)
        opt_k = 9

        section_calculate_optimal_k = False
        if section_calculate_optimal_k:
            k_range = (3, 30)
            sse = pd.Series(index=range(k_range[0], k_range[1]))
            silhouette = pd.Series(index=range(k_range[0], k_range[1]))

            for k in tqdm(range(k_range[0], k_range[1]), desc='Calculating optimal K'):
                model = KMeans(init="random", n_clusters=k, n_init=10, max_iter=300, random_state=42)
                model.fit(scaled_X)
                sse[k] = model.inertia_
                silhouette[k] = silhouette_score(scaled_X, model.labels_)

            kl = KneeLocator(range(k_range[0], k_range[1]), sse, curve="convex", direction="decreasing")
            opt_k = kl.elbow + 3

            fig, axs = plt.subplots(2)
            fig.suptitle('Vertically stacked subplots')
            axs[0].plot(range(k_range[0], k_range[1]), sse)
            _ = axs[0].vlines(x=opt_k, ymin=sse.min() * 0.5, ymax=sse.max() * 1.2, colors='r')
            axs[0].set_title('SSE')

            axs[1].plot(range(k_range[0], k_range[1]), silhouette)
            axs[1].set_title('silhouette')
            _ = axs[1].vlines(x=opt_k, ymin=silhouette.min() * 0.5, ymax=silhouette.max() * 1.2, colors='r')

            plt.style.use("fivethirtyeight")
            plt.xticks(range(k_range[0], k_range[1]))
            # plt.show()
            figpath = os.path.join(self.outpath, 'optimal K for clustering')
            plt.savefig(figpath)

        print(f'Types of voters: {opt_k}')
        model = KMeans(init="random", n_clusters=opt_k, n_init=10, max_iter=300, random_state=42)
        model.fit(scaled_X)
        self.votes_dist_df['Label'] = model.labels_
        self.votes_dist_df['Votes'] = self.votesdf.sum(axis=1).values
        self.votesdf['Label'] = self.votes_dist_df['Label'].values

        self.votes_dist_df = self.votes_dist_df.sort_values(by='Label', ascending=False)
        voters = self.votes_dist_df[prts + ['Label']].groupby('Label').mean()
        voters['Votes'] = self.votes_dist_df.groupby('Label')['Votes'].sum()
        voters['Votes ratio'] = voters['Votes'] / voters['Votes'].sum()

        cols_to_norm = list(self.non_empy_parties.index) + ['Votes ratio']
        # voters[cols_to_norm] *= 100
        # voters[cols_to_norm] = np.round(voters[cols_to_norm], 1)
        voters[cols_to_norm] = np.round(voters[cols_to_norm], 3)

        for cls in range(opt_k):
            xdf = self.votes_dist_df[self.votes_dist_df['Label'] == cls]
            t = xdf.iloc[:6].index.to_list()
            print(f'--- {cls} ---')
            for tq in t:
                print(tq)

        nonempties = voters.max() > 0
        nonempties = nonempties[nonempties]
        voters = voters[nonempties.index]

        for c in self.metadata:
            self.votesdf[c] = 0
        self.votesdf.loc[:, self.metadata] = self.rawdf.loc[self.votesdf.index, self.metadata]

        self.voters = voters
        self.add_external_data()

    def vote_sampling(self, n=50):
        if type(n) is not tuple:
            n = (n, 1)
        n_votes = n[0]
        n_samples = n[1]
        samples = np.random.choice(self.norm_count.index, p=self.norm_count.values, size=n)
        df = pd.DataFrame(index=range(n_votes), columns=range(n_samples), data=samples)
        v = df.apply(pd.value_counts).fillna(0).astype(int).T

        for c in self.parties:
            if c not in v.columns:
                v[c] = 0
        return v

    def optimize(self):
        v = self.vote_sampling((100, 1000))

        s = v.sum(axis=1)

        parties = len(self.parties)
        voters = self.meatavoters
        votes_per_voter = 3

        votes = pd.DataFrame(index=[f'a{i}' for i in range(voters)], columns=self.parties, data=0)
        for voter_i, party_i in enumerate(self.norm_count.head(voters).index.to_list()):
            votes.loc[f'a{voter_i}', party_i] = 1.0

    def export(self):
        for df, title in [
            (self.votes_dist_df, 'votes distributions'),
            (self.voters, 'Meta voters'),
            (self.votesdf, 'votes unnormalized'),
        ]:
            if df is not None:
                df.to_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        # Towns by label
        df = self.votesdf[['Label', 'שם ישוב']]
        tdf = pd.DataFrame(columns=sorted(df['Label'].unique()), index=range(df['Label'].value_counts().max()))
        for c in df['Label'].unique():
            r = df[df['Label'].eq(c)]['שם ישוב'].values
            tdf.loc[range(len(r)), c] = df[df['Label'].eq(c)]['שם ישוב'].values
        title = 'Towns cluster'
        tdf.to_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')


if __name__ == '__main__':
    root_path = os.path.dirname(__file__)
    data_path = os.path.join(root_path, 'data')

    df = pd.read_csv(data_path + '\\' + 'votes per settlement 2020.csv', encoding='iso_8859_8')

    voter = Voter(df)
    voter.cluster()
    voter.export()
    print("END OF CODE.")