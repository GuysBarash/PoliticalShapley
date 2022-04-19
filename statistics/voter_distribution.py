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

from sklearn.metrics.pairwise import euclidean_distances


def ballots_to_json(path_in, path_out):
    with open(path_in, encoding="utf8") as fp:
        r = fp.read()
    r = r.split('\n')
    d = {re.search(r'([^\s]+)\s(.*)', t).group(1): re.search(r'([^\s]+)\s(.*)', t).group(2) for t in r}
    with open(path_out, 'w', encoding='utf-8') as file:
        json.dump(d, file, ensure_ascii=False, indent=4)


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
    def __init__(self):
        self.rootpath = os.path.dirname(__file__)
        self.outpath = os.path.join(self.rootpath, 'Results')
        self.datapath = os.path.join(self.rootpath, 'data')
        self.lms_path = os.path.join(self.datapath, 'lms')
        clear_folder(self.outpath, delete_if_exist=False)
        clear_folder(self.rootpath, delete_if_exist=False)
        clear_folder(self.lms_path, delete_if_exist=False)

        self.rawdf = None
        self.ballots = None
        self.metadata = None
        self.parties = None
        self.voters = None
        self.town_clusters = None
        self.votesdf = None
        self.votes_dist_df = None
        self.non_empy_parties = None
        self.distances = None
        self.k = None

    def initialize(self, votes_per_settelments_path, ballots_path=None):
        df = pd.read_csv(votes_per_settelments_path, encoding='iso_8859_8')
        self.rawdf = df.copy()
        self.rawdf = self.rawdf[~self.rawdf['שם ישוב'].eq("מעטפות חיצוניות")]
        self.rawdf.loc[:, 'uid'] = self.rawdf.loc[:, 'סמל ישוב'].astype(int).values

        self.ballots = None
        if ballots_path is not None:
            with open(ballots_path, encoding='utf-8') as json_file:
                self.ballots = json.load(json_file)
            self.rawdf = self.rawdf.rename(columns=self.ballots)

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

    def add_external_data(self):
        cols_to_mean = list()
        cols_to_most_common = list()

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

        section_age = True
        if section_age:
            age = pd.read_csv(os.path.join(self.lms_path, 'age.csv'), encoding='utf-8-sig')
            age = age[~age['uid'].isna()]

            age_bins = [c for c in age.columns if c not in ['town', 'uid', 'total population']]
            q = age[[c for c in age.columns if c not in ['town']]]
            for c in ['age 0 to 5', 'age 5 to 18', 'age 19 to 45', 'age 46 to 55', 'age 56 to 64', 'age 65 and above']:
                q[c] = q[c].astype(float) / q['total population'].astype(float)
            self.votesdf = self.votesdf.merge(q, left_on='uid', right_on='uid', how='outer')
            cols_to_mean += age_bins + ['total population']

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

        section_crime = True
        if section_crime:
            crime = pd.read_csv(os.path.join(self.lms_path, 'police_info.csv'), encoding='utf-8-sig')
            for t in crime.columns:
                if 'Unnamed' in t:
                    crime = crime.drop(t, axis=1)

            cols = ['Crime public security', 'Crime moral', 'Crime property',
                    'Crime sex', 'Crime cheat', 'Crime people', 'Crime body',
                    'Crime public order', 'Crime license', 'Crime financial',
                    'Crime driving', 'Crime beuracracy', 'Crime of definition',
                    'Crime other']
            q = pd.DataFrame(index=crime.index)
            q['uid'] = crime['uid'].astype(int)
            q['Crime public security'] = crime['עבירות בטחון'].astype(int)
            q['Crime moral'] = crime['עבירות כלפי המוסר'].astype(int)
            q['Crime property'] = crime['עבירות כלפי הרכוש'].astype(int)
            q['Crime sex'] = crime['עבירות מין'].astype(int)
            q['Crime cheat'] = crime['עבירות מרמה'].astype(int)
            q['Crime people'] = crime['עבירות נגד אדם'].astype(int)
            q['Crime body'] = crime['עבירות נגד גוף'].astype(int)
            q['Crime public order'] = crime['עבירות סדר ציבורי'].astype(int)
            q['Crime license'] = crime['עבירות רשוי'].astype(int)
            q['Crime financial'] = crime['עבירות כלכליות'].astype(int)
            q['Crime driving'] = crime['עבירות תנועה'].astype(int)
            q['Crime beuracracy'] = crime['עבירות מנהליות'].astype(int)
            q['Crime of definition'] = crime['סעיפי הגדרה'].astype(int)
            q['Crime other'] = crime['שאר עבירות'].astype(int)
            q['Crimes total'] = q[cols].sum(axis=1)
            self.votesdf = self.votesdf.merge(q, left_on='uid', right_on='uid', how='outer')

            cols_to_normal = cols + ['Crimes total']
            cols_to_mean += cols + ['Crimes total']
            for c in cols_to_normal:
                self.votesdf[c] = (100 * self.votesdf[c].astype(float)) / self.votesdf['population']

        section_mosques = True
        if section_mosques:
            reldf = pd.read_csv(os.path.join(self.lms_path, 'Mosques.csv'), encoding='utf-8-sig', index_col=0)

            self.votesdf = self.votesdf.merge(reldf, left_on='uid', right_on='uid', how='outer')
            cols_to_mean += ['Mosques']
            cols_to_normal = ['Mosques']
            for c in cols_to_normal:
                self.votesdf[c] = (100 * self.votesdf[c].astype(float)) / self.votesdf['population'].astype(float)

        self.votesdf = self.votesdf[~self.votesdf['Label'].isna()]
        self.voters['Leading party'] = self.voters[[c for c in self.voters.columns if c in self.parties]].idxmax(
            axis=1)

        section_add_means = True
        if section_add_means:
            cols_to_mean += ['Jews', 'Arab-Muslim', 'Arab-Christian', 'Druze',
                             'Bagrut', 'Degree',
                             'under 18', 'above 75', 'natural growth', 'density',
                             'Salary', 'JINI',
                             'population', 'Jews pop', 'Non-Jews pop',
                             ]
            for c in cols_to_mean:
                if c not in self.votesdf.columns:
                    continue
                df = self.votesdf[['Label', c]]
                g = df.groupby('Label')
                nullratio = g.agg({c: lambda x: x.isnull().mean()})
                nullratio = nullratio[nullratio < 0.6].dropna()
                self.voters.loc[nullratio.index.astype(int).to_list(), c] = g[c].median()

        section_add_most_common = True
        if section_add_most_common:
            cols_to_most_common += ['county', 'Settelment type'
                                    ]
            for c in cols_to_most_common:
                if c not in self.votesdf.columns:
                    continue
                df = self.votesdf[['Label', c]]
                g = df.groupby('Label')

                nullratio = g.agg({c: lambda x: x.isnull().mean()})
                nullratio = nullratio[nullratio < 0.5].dropna()
                self.voters.loc[nullratio.index.astype(int).to_list(), c] = g[c].agg(
                    lambda x: x.value_counts().index[0])

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
            opt_k = kl.elbow + 1

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
        self.k = opt_k
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

        print('<-------------------------->')
        for cls in range(opt_k):
            xdf = self.votes_dist_df[self.votes_dist_df['Label'] == cls]
            t = xdf.iloc[:6].index.to_list()
            print(f'--- {cls} ---')
            for tq in t:
                print(tq)
        print('<-------------------------->')

        nonempties = voters.max() > 0
        nonempties = nonempties[nonempties]
        voters = voters[nonempties.index]

        for c in self.metadata:
            self.votesdf[c] = 0
        self.votesdf.loc[:, self.metadata] = self.rawdf.loc[self.votesdf.index, self.metadata]

        self.voters = voters
        self.voters['Towns count'] = self.votesdf.groupby('Label')['uid'].count()
        self.add_external_data()

    def export(self):
        for df, title in [
            (self.votes_dist_df, 'votes distributions'),
            (self.voters, 'Meta voters'),
            (self.votesdf, 'votes unnormalized'),
            (self.rawdf, 'Raw data'),
            (self.distances, 'town distances'),

        ]:
            if df is not None:
                df.to_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        # Towns by label
        df = self.votesdf[['Label', 'שם ישוב']]
        tdf = pd.DataFrame(columns=sorted(df['Label'].unique()), index=range(df['Label'].value_counts().max()))
        for c in df['Label'].unique():
            r = df[df['Label'].eq(c)]['שם ישוב'].values
            tdf.loc[range(len(r)), c] = df[df['Label'].eq(c)]['שם ישוב'].values

        self.town_clusters = tdf
        title = 'Towns cluster'
        tdf.to_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'notes signs'
        with open(os.path.join(self.outpath, f'{title}.json'), 'w', encoding='utf-8') as file:
            json.dump(self.ballots, file, ensure_ascii=False, indent=4)

        # Additional information
        info = dict()
        info['metadata'] = self.metadata
        info['parties'] = self.parties
        info['non_empy_parties'] = self.non_empy_parties.index.to_list()
        info['k'] = self.k
        title = 'additional info'
        with open(os.path.join(self.outpath, f'{title}.json'), 'w', encoding='utf-8') as file:
            json.dump(info, file, ensure_ascii=False, indent=4)

    def stats_import(self):
        s_time = datetime.datetime.now()

        title = 'votes distributions'
        self.votes_dist_df = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'Meta voters'
        self.voters = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'votes unnormalized'
        self.votesdf = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'Towns cluster'
        self.town_clusters = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'Raw data'
        self.rawdf = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'town distances'
        self.distances = pd.read_csv(os.path.join(self.outpath, f'{title}.csv'), encoding='utf-8-sig')

        title = 'notes signs'
        with open(os.path.join(self.outpath, f'{title}.json'), encoding='utf-8-sig') as file:
            self.ballots = json.load(file)

        title = 'additional info'
        with open(os.path.join(self.outpath, f'{title}.json'), encoding='utf-8-sig') as file:
            info = json.load(file)
            self.metadata = info['metadata']
            self.parties = info['parties']
            self.k = info['k']
            non_empy_parties = info['non_empy_parties']
            self.non_empy_parties = pd.Series(index=non_empy_parties, data=True)

        e_time = datetime.datetime.now()
        d_time = e_time - s_time
        print(f'Data load completed. Time: {d_time}')


if __name__ == '__main__':
    root_path = os.path.dirname(__file__)
    data_path = os.path.join(root_path, 'data')

    votes_csv_path = data_path + '\\' + 'votes per settlement 2021.csv'
    ballots_path = r"C:\school\PoliticalShapley\statistics\data\votes per settlement 2021 ballots.json"

    voter = Voter()
    voter.initialize(votes_csv_path, ballots_path=ballots_path)
    voter.cluster()
    voter.export()
    print("END OF CODE.")
