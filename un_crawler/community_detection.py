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


class UNcrawler:
    @staticmethod
    def get_driver():
        driverpath = os.path.join(os.path.dirname(__file__), "chromedriver.exe")
        if not os.path.exists(driverpath):
            driverpath = r"C:\school\PoliticalShapley\Knesset_crawler\chromedriver.exe"
        driver = webdriver.Chrome(driverpath)
        driver.wait = WebDriverWait(driver, 5)
        time.sleep(1)
        return driver

    @staticmethod
    def crawl_single_resolution(pckg):
        url = pckg['link']

        ret = dict()
        ret.update(pckg)
        json_path = ret['json']

        if os.path.exists(json_path):
            ret['skipped'] = True
            return ret
        ret['skipped'] = False

        try:
            driver = UNcrawler.get_driver()
            driver.get(url)
            time.sleep(1)
            html = driver.page_source
            soup = BeautifulSoup(html, 'lxml')
            rows = soup.find_all('div', attrs={'class': 'metadata-row'})
            ret['rows count'] = len(rows)

            def clean_str(s):
                s = re.sub('[\n]+', '\n', s)
                s = re.sub('^\n', '', s)
                return s

            for ridx, row in enumerate(rows):
                title_node, value_node = list(row.children)
                title = title_node.text
                txt = value_node.get_text(separator="\n")
                ret[f'title_{ridx}'] = clean_str(title)
                ret[f'text_{ridx}'] = clean_str(txt)

            driver.quit()
            with open(json_path, "w") as outfile:
                json.dump(ret, outfile, indent=4)
            ret['ERROR'] = None
        except Exception as e:
            ret['ERROR'] = str(e)
            try:
                driver.quit()
            except Exception as e:
                pass

        return ret

    @staticmethod
    def crawl_single_json(pckg):
        r_idx = pckg['token']
        json_path = pckg['jpath']

        ret = dict()
        ret['token'] = r_idx
        ret['ERROR'] = None

        if not os.path.exists(json_path):
            ret['MISSING FILE'] = True
            ret['ERROR'] = 'MISSING FILE'
            print(f"FILE NOT DOWNLOADED: {r_idx}")
            return ret
        ret['MISSING FILE'] = False
        ret['MISSING DATE'] = False
        ret['MISSING RES_TITLE'] = False
        ret['MISSING VOTES'] = False
        ret['MISSING LINES JSON'] = False
        ret['BAD VOTE TOKEN'] = False

        try:
            with open(json_path) as fp:
                d = json.load(fp)

            token = r_idx
            date = None
            res_title = None
            draft_res_title = None
            vots_s = None
            agenda_id = None
            council = None
            resolution_description = None

            items = d.get('rows count', 0)
            if items == 0:
                ret['MISSING LINES JSON'] = True
                ret['ERROR'] = f'MISSING LINES JSON'
                os.remove(json_path)
                print(f"FILE BAD FORMAT: {r_idx} << REMOVING")
                return ret

            for i in range(items):
                title = d[f'title_{i}'].replace('\n', '')
                txt = d[f'text_{i}']
                if title == 'Agenda':
                    agenda_id = txt.replace('\n', '')
                elif title in ['Resolution']:
                    res_title = txt.replace('\n', '')

                elif title in ['Draft resolution']:
                    draft_res_title = txt.replace('\n', '')

                elif title in ['Vote', 'Votes']:
                    vots_s = txt

                elif title in ['Vote date']:
                    date = txt

                elif title in ['Title']:
                    resolution_description = txt

                elif title in ['Collections']:
                    council = re.search(r'UN Bodies\n > \n([^\n]+)\n', txt)
                    if council is not None:
                        council = council.group(1)

                else:
                    pass

            j = 3
            if date is None:
                saved = False
                special_cases = {642010: '1977', 724807: '2011'}
                special_date = special_cases.get(r_idx, None)
                if special_date is not None:
                    date = special_date
                    saved = True

                if not saved:
                    ret['MISSING DATE'] = True
                    ret['ERROR'] = 'MISSING DATE'
                    return ret
            if res_title is None:
                saved = False
                special_cases = {3836031: 'MISSING_DATA'}
                special_title = special_cases.get(r_idx, None)
                if draft_res_title is not None:
                    res_title = draft_res_title
                    saved = True
                elif special_title is not None:
                    res_title = special_title
                    saved = True

                if not saved:
                    ret['MISSING RES_TITLE'] = True
                    ret['ERROR'] = 'MISSING RES_TITLE'
                    return ret
            if vots_s is None:
                ret['MISSING VOTES'] = True
                ret['ERROR'] = 'MISSING VOTES'
                return ret
            if council is None:
                # Try to fetch from title
                saved = False
                if resolution_description is not None:
                    if 'General Assembly' in resolution_description:
                        saved = True
                        council = 'General Assembly'

                if not saved:
                    special_cases = {3949880: 'General Assembly', 3949879: 'General Assembly'}
                    council = special_cases.get(r_idx, None)
                    if council is not None:
                        saved = True

                if not saved:
                    print(f"BAD ASSEMBLY AT: {r_idx}")
                    ret['MISSING COUNCIL'] = True
                    ret['ERROR'] = 'MISSING COUNCIL'
                    return ret

            ret['date'] = date
            ret['title'] = res_title
            ret['council'] = council
            votes = list()
            votes_lst = [re.sub('^[\s]+', '', vt) for vt in vots_s.split('\n')]
            sr = pd.Series()
            for v in votes_lst:
                m = re.search(r'^([^\s])\s', v)
                if m is not None:
                    m = m.group(1).upper()
                    if m not in ['A', 'N', 'Y']:
                        ret['BAD VOTE TOKEN'] = True
                        ret['ERROR'] = f'BAD VOTE TOKEN - {m}'
                        return ret
                    country = v[2:]
                else:
                    country = v
                    m = 'X'

                sr[country] = m

            ret['votes'] = sr
            ret['ERROR'] = None
        except Exception as e:
            ret['ERROR'] = str(e)

        return ret

    def __init__(self):
        self.root_path = os.path.dirname(__file__)
        self.outdir = os.path.join(self.root_path, 'data')
        self.json_dir = os.path.join(self.outdir, 'Resolutions jsons')
        self.json_df_path = os.path.join(self.json_dir, 'json_df.csv')
        self.reslinksdf_path = os.path.join(self.outdir, 'Links.csv')
        self.final_db_path = os.path.join(self.outdir, 'UN DATA.csv')

        clear_folder(self.root_path, False)
        clear_folder(self.outdir, False)
        clear_folder(self.json_dir, False)

        self.reslinksdf = None

        self.search_base = r'https://digitallibrary.un.org/search?c=Voting+Data&jrec=@C@&cc=Voting+Data&ln=en&rg=100&fct__3=@Y@&fct__9=Vote&fct__9=Vote&sf=year'
        self.search_range = 3
        self.start_year = 1946
        self.end_year = datetime.datetime.now().year

        # options = Options()
        # options.add_argument('--remote-debugging-port=9222')
        # options.add_experimental_option("useAutomationExtension", False)
        # options.set_capability("acceptInsecureCerts", True)
        # options.set_capability("acceptSslCerts", True)
        # options.add_experimental_option("excludeSwitches", ["enable-automation"])
        # options.add_experimental_option('excludeSwitches', ['enable-logging'])
        # options.add_experimental_option('useAutomationExtension', False)
        # options.add_argument('--disable-blink-features=AutomationControlled')

        print("driver initialized")

    def search_records(self):
        if os.path.exists(self.reslinksdf_path):
            self.reslinksdf = pd.read_csv(self.reslinksdf_path)
            self.reslinksdf['idx'] = self.reslinksdf['uid']
            self.reslinksdf = self.reslinksdf.set_index('idx')
            print("Links already exist. Skipping.")
        else:
            self.driver = UNcrawler.get_driver()
            self.reslinksdf = pd.DataFrame(columns=['Name', 'Year', 'uid', 'page', 'link'])
            fetched = dict()
            for year in range(self.start_year, self.end_year):
                for i in tqdm(range(self.search_range), f'Searching Res links [Year: {year}]'):
                    c = (i * 100) + 1
                    c_url = deepcopy(self.search_base)
                    c_url = c_url.replace('@C@', str(c))
                    c_url = c_url.replace('@Y@', str(year))
                    time.sleep(5)
                    # print(f"Scanning link: {c_url}")
                    self.driver.get(c_url)
                    time.sleep(1)
                    html = self.driver.page_source
                    soup = BeautifulSoup(html, 'lxml')
                    links = soup.find_all('a')
                    for idx, l in enumerate(links):
                        if not l.has_attr('href'):
                            pass
                        else:
                            href = l['href']
                            m = re.search('record/([0-9]+)\?', href)
                            if m is not None:
                                uid = int(m.group(1))
                                if fetched.get(uid, None) is not None:
                                    continue
                                fetched[uid] = href
                                href_full = r'https://digitallibrary.un.org' + href
                                name = l.text.replace('\n', '')
                                self.reslinksdf.loc[uid] = [name, year, uid, href, href_full]
                time.sleep(0.1)
                print(f"{self.reslinksdf[self.reslinksdf['Year'].eq(year)].shape[0]} Resolutions found for year {year}")
                print(f"{self.reslinksdf.shape[0]} Resolutions total.")
                time.sleep(0.1)
                self.reslinksdf.to_csv(self.reslinksdf_path, encoding='utf-8-sig', index=False)
                self.driver.quit()

    def scrap_all_resolutions(self, parallel=False):
        inputs = list()
        for ridx, r in self.reslinksdf.iterrows():
            pckg = dict()
            pckg['idx'] = ridx
            pckg['outpath'] = self.json_dir
            pckg['json'] = os.path.join(pckg['outpath'], f'{str(pckg["idx"])}.json')
            pckg.update(r.to_dict())
            if os.path.exists(pckg['json']):
                pass
            else:
                inputs.append(pckg)

        missed = 0
        hits = 0
        if parallel:
            pool = ThreadPool()
            res = pool.imap_unordered(func=UNcrawler.crawl_single_resolution, iterable=inputs)
            time.sleep(0.1)
            for idx in tqdm(range(len(inputs)), desc='Scanning Resolutions [Parallel]'):
                outp_pckg = res.next()
                if outp_pckg['ERROR'] is not None:
                    missed += 1
                else:
                    hits += 1
            time.sleep(0.1)
            pool.close()
            pool.join()
            time.sleep(0.1)
        else:
            for inp_pckg in tqdm(inputs, desc='Scanning Resolutions [Concurrent]'):
                outp_pckg = UNcrawler.crawl_single_resolution(inp_pckg)
                if outp_pckg['ERROR'] is not None:
                    missed += 1
                else:
                    hits += 1

        total_attempts = hits + missed + 0.1
        hit_ratio = 100.0 * hits / total_attempts
        miss_ratio = 100.0 * missed / total_attempts
        print(f"Scrapping completed.")
        print(f'Hits: {hits}/{total_attempts} ({hit_ratio:>.1f})')
        print(f'Miss: {missed}/{total_attempts} ({miss_ratio:>.1f})')

    def investigate_jsons(self, parallel=False):
        checked = 0
        hits = 0
        missed_meta = 0
        missed_votes = 0
        vdf = pd.DataFrame(columns=['Council', 'Date', 'Resolution', 'token'])
        if os.path.exists(self.json_df_path):
            vdf = pd.read_csv(self.json_df_path, index_col=0, low_memory=False)

        inputs = list()
        for r_idx in list(set(self.reslinksdf.index) - set(vdf.index)):
            pckg = dict()
            jpath = os.path.join(self.json_dir, f'{str(r_idx)}.json')
            pckg['jpath'] = jpath
            pckg['token'] = r_idx
            inputs.append(pckg)

        no_votes = 0
        hits = 0
        invalid_file = 0
        file_not_downloaed = 0
        empty_json = 0
        if parallel:
            pool = ThreadPool()
            res = pool.imap_unordered(func=UNcrawler.crawl_single_json, iterable=inputs)
            time.sleep(0.1)
            for idx in tqdm(range(len(inputs)), desc='Scanning Resolutions JSON [Parallel]'):
                outp_pckg = res.next()
                r_idx = outp_pckg['token']
                if outp_pckg['ERROR'] is not None:
                    if outp_pckg.get('MISSING VOTES'):
                        no_votes += 1
                    elif outp_pckg.get('MISSING FILE'):
                        file_not_downloaed += 1
                    elif outp_pckg.get('MISSING LINES JSON'):
                        empty_json += 1

                    else:
                        invalid_file += 1
                        print(f"BAD JSON AT {r_idx}")
                        pass
                else:
                    hits += 1
                    sr = outp_pckg['votes']
                    for c in list(set(sr.index) - set(vdf.columns)):
                        vdf[c] = np.NaN
                    vdf.loc[r_idx, sr.index] = sr
                    vdf.loc[r_idx, ['Council', 'Date', 'Resolution', 'token']] = outp_pckg['council'], outp_pckg[
                        'date'], outp_pckg['title'], outp_pckg['token']
            time.sleep(0.1)
            pool.close()
            pool.join()
            time.sleep(0.1)
        else:
            for inp_pckg in tqdm(inputs, desc='Scanning Resolutions JSON [Concurrent]'):
                outp_pckg = UNcrawler.crawl_single_json(inp_pckg)
                r_idx = outp_pckg['token']
                if outp_pckg['ERROR'] is not None:
                    if outp_pckg.get('MISSING VOTES'):
                        no_votes += 1
                    elif outp_pckg.get('MISSING FILE'):
                        file_not_downloaed += 1
                    elif outp_pckg.get('MISSING LINES JSON'):
                        empty_json += 1

                    else:
                        invalid_file += 1
                        print(f"BAD JSON AT {r_idx}")
                        pass
                else:
                    hits += 1
                    sr = outp_pckg['votes']
                    for c in list(set(sr.index) - set(vdf.columns)):
                        vdf[c] = np.NaN
                    vdf.loc[r_idx, sr.index] = sr
                    vdf.loc[r_idx, ['Council', 'Date', 'Resolution', 'token']] = outp_pckg['council'], outp_pckg[
                        'date'], outp_pckg['title'], outp_pckg['token']

        print(f"Hits: {hits}")
        print(f"Files not downloaded: {file_not_downloaed}")
        print(f"BAD FILE: {invalid_file}")
        print(f"EMPTY json FILE: {empty_json}")
        print(f"Missed votes: {no_votes}")
        vdf.to_csv(self.json_df_path)

    def finalize(self):
        votes_df = pd.read_csv(self.json_df_path, index_col=0, low_memory=False)
        linksdf = pd.read_csv(self.reslinksdf_path, index_col=0, low_memory=False, encoding='utf-8-sig')

        linksdf['Title'] = linksdf.index
        linksdf = linksdf.set_index('uid')

        commonidx = list(votes_df.index)
        votes_df.insert(2, 'Title', linksdf.loc[commonidx, 'Title'])
        votes_df.insert(4, 'Link', linksdf.loc[commonidx, 'link'])

        votes_df['token'] = votes_df['token'].astype(int)

        data_cols = ['Council', 'Date', 'Title', 'Resolution', 'Link', 'token']
        country_cols = [c for c in votes_df.columns if c not in data_cols]

        votes_df.insert(4, 'YES COUNT', votes_df[country_cols].eq('Y').sum(axis=1))
        votes_df.insert(4, 'NO COUNT', votes_df[country_cols].eq('N').sum(axis=1))
        votes_df.insert(4, 'ABSENT COUNT', votes_df[country_cols].eq('A').sum(axis=1))
        votes_df.insert(4, 'NO-VOTE COUNT', votes_df[country_cols].eq('X').sum(axis=1))
        votes_df.insert(4, 'TOTAL VOTES', (~votes_df[country_cols].isna()).sum(axis=1))
        votes_df.to_csv(self.final_db_path, encoding='utf-8-sig', index=False)
        print(f"Database exported to: {self.final_db_path}")

        # KAggle
        undf = votes_df
        undf['YEAR'] = undf['Date'].str.extract(r'([0-9]{4})').astype(int)

        metadf = votes_df[country_cols].T
        yes_votes = metadf.eq('Y').sum(axis=1)
        no_votes = metadf.eq('N').sum(axis=1)
        abs_votes = metadf.eq('A').sum(axis=1)
        missed_votes = metadf.eq('X').sum(axis=1)
        metadf.insert(0, 'DIDNT_VOTE', missed_votes)
        metadf.insert(0, 'ABS_VOTES', abs_votes)
        metadf.insert(0, 'NO_VOTES', no_votes)
        metadf.insert(0, 'YES_VOTES', yes_votes)

        for c in country_cols:
            s = votes_df[c]
            tokens = s.unique()
            new_tokens = [t for t in tokens if t not in ['Y', 'A', 'N', np.nan, 'X']]
            if len(new_tokens):
                print(f"Country: {c}\tToken: {new_tokens}")


if __name__ == '__main__':

    section_communities = False
    if section_communities:
        gundf = pd.read_csv(r"C:\school\PoliticalShapley\un_crawler\data\UN DATA.csv", low_memory=False)
        vdf_path = os.path.join(r"C:\school\PoliticalShapley\un_crawler\data", 'similar votes.csv')
        gundf.insert(2, 'YEAR', gundf['Date'].str.extract(r'([0-9]{4})').astype(int))

        start_year = 1950
        step = 3
        window = 5
        mx_year = gundf['YEAR'].max()
        windows = [(s, s + window) for s in range(start_year, mx_year, step) if s + window <= mx_year]
        for idx, (s_time, e_time) in enumerate(windows):
            undf = gundf[gundf['YEAR'] >= s_time]
            undf = undf[undf['YEAR'] < e_time]
            undf = undf[undf['Council'].eq('General Assembly')]

            metacols = undf.columns[:12].to_list()
            statecols = [c for c in undf.columns if c not in metacols]

            gadf = undf[statecols]
            vdf = pd.DataFrame(index=statecols, columns=statecols)
            for from_state in tqdm(statecols, desc=f'Building adjacency ({s_time}-->{e_time})'):
                f_df = gadf[from_state].fillna('FNA')
                t_df = gadf.fillna('TNA')
                e = (t_df.T == f_df).sum(axis=1)
                vdf.loc[from_state] = e
                vdf.loc[from_state, from_state] = (~f_df.eq('FNA')).sum()

            minimal_value = 10
            empties = vdf.sum(axis=1) <= minimal_value
            states = empties[~empties].index.tolist()
            vdf = vdf.loc[states, states]

            min_votes = 50
            max_edges_per_state = 20

            vdf = vdf[vdf.max(axis=1) > min_votes]
            vdf = vdf[list(vdf.index)]
            nvdf = vdf / vdf.max(axis=1)
            G = nx.from_pandas_adjacency(nvdf, create_using=nx.DiGraph)
            for c in nvdf.columns:
                nvdf.loc[c, c] = 0
                z = nvdf.sort_values(by=c, ascending=False)[c].iloc[max_edges_per_state:].index.to_list()
                nvdf.loc[c, z] = 0

            reduced_nvdf = nvdf.copy()
            for c in reduced_nvdf.columns:
                reduced_nvdf.loc[c, c] = 0
                z = reduced_nvdf.sort_values(by=c, ascending=False)[c].iloc[15:].index.to_list()
                reduced_nvdf.loc[c, z] = 0
            G = nx.from_pandas_adjacency(reduced_nvdf, create_using=nx.DiGraph)

            section_girvan_newman = True
            if section_girvan_newman:
                k = 1
                print("Calculating")
                set2key = lambda subset: ';'.join(sorted(list(subset)))
                comp = community.girvan_newman(G)
                communities_t = [t for t in tqdm((itertools.islice(comp, k, k + 20)), desc='Bulding communities')]
                single_states = reduced_nvdf.index.to_list()
                collective_group = set(single_states)
                # building initial dict of node_id to each possible subset:
                node_id = 0
                init_node2community_dict = dict()  # {node_id: communities_t[0][0].union(communities_t[0][1])}
                individual_community = [set([s]) for s in single_states]
                communities = [tuple([collective_group])] + communities_t + [individual_community]
                for comm in communities:
                    for subset in list(comm):
                        if subset not in init_node2community_dict.values():
                            init_node2community_dict[node_id] = subset
                            node_id += 1

                idx2community = {set2key(v): k for k, v in init_node2community_dict.items()}
                # turning this dictionary to the desired format in @mdml's answer
                node_id_to_children = {e: [] for e in init_node2community_dict.keys()}
                for comm_level_idx in range(len(communities) - 1):
                    comm_level_parent = communities[comm_level_idx]
                    comm_level_child = communities[comm_level_idx + 1]
                    for parent in comm_level_parent:
                        parent_idx = idx2community[set2key(parent)]
                        for child in comm_level_child:
                            intersection_size = len(child.intersection(parent))
                            if intersection_size == 0:
                                pass
                            else:
                                child_idx = idx2community[set2key(child)]
                                if child_idx != parent_idx:
                                    node_id_to_children[parent_idx] = node_id_to_children[parent_idx] + [child_idx]

                # also recording node_labels dict for the correct label for dendrogram leaves
                node_labels = dict()
                for node_id, group in init_node2community_dict.items():
                    if len(group) == 1:
                        node_labels[node_id] = list(group)[0]
                    else:
                        node_labels[node_id] = ''

                section_dendogram = False
                if section_dendogram:
                    # also needing a subset to rank dict to later know within all k-length merges which came first
                    subset_rank_dict = dict()
                    rank = 0
                    for e in communities[::-1]:
                        for p in list(e):
                            if tuple(p) not in subset_rank_dict:
                                subset_rank_dict[tuple(sorted(p))] = rank
                                rank += 1
                    subset_rank_dict[tuple(sorted(itertools.chain.from_iterable(communities[-1])))] = rank


                    # my function to get a merge height so that it is unique (probably not that efficient)
                    def get_merge_height(sub):
                        sub_tuple = tuple(sorted([node_labels[i] for i in sub]))
                        n = len(sub_tuple)
                        other_same_len_merges = {k: v for k, v in subset_rank_dict.items() if len(k) == n}
                        if len(other_same_len_merges) == 0:
                            range = 1
                            min_rank, max_rank = 1, 1
                        else:
                            min_rank, max_rank = min(other_same_len_merges.values()), max(
                                other_same_len_merges.values())
                            range = (max_rank - min_rank) if max_rank > min_rank else 1
                        return float(len(sub)) + 0.8 * (subset_rank_dict[sub_tuple] - min_rank) / range


                    # finally using @mdml's magic, slightly modified:
                    Gt = nx.DiGraph(node_id_to_children)
                    nodes = Gt.nodes()
                    leaves = set(n for n in nodes if Gt.out_degree(n) == 0)
                    inner_nodes = [n for n in nodes if Gt.out_degree(n) > 0]

                    # Compute the size of each subtree
                    subtree = dict((n, [n]) for n in leaves)
                    for u in inner_nodes:
                        children = set()
                        node_list = list(node_id_to_children[u])
                        while len(node_list) > 0:
                            v = node_list.pop(0)
                            children.add(v)
                            node_list += node_id_to_children[v]
                        subtree[u] = sorted(children & leaves)

                    inner_nodes.sort(
                        key=lambda n: len(subtree[n]))  # <-- order inner nodes ascending by subtree size, root is last

                    # Construct the linkage matrix
                    leaves = sorted(leaves)
                    index = dict((tuple([n]), i) for i, n in enumerate(leaves))
                    Z = []
                    k = len(leaves)
                    for i, n in enumerate(inner_nodes):
                        children = node_id_to_children[n]
                        x = children[0]
                        for y in children[1:]:
                            z = tuple(sorted(subtree[x] + subtree[y]))
                            i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]

                            subtree_n = subtree[n]
                            merge_height = get_merge_height(subtree_n)
                            zval = [i, j, merge_height, len(z)]
                            Z.append(zval)  # <-- float is required by the dendrogram function
                            index[z] = k
                            subtree[z] = list(z)
                            x = z
                            k += 1

                    # dendrogram
                    from scipy.cluster.hierarchy import dendrogram

                    fig1, ax1 = plt.subplots(1, figsize=(15, 35))
                    labels = [node_labels[node_id] for node_id in leaves]
                    dendrogram(Z, orientation='right', labels=labels, ax=ax1)
                    plt.savefig('dendrogram.png')
                    plt.close()

                top_a = 10

                idx = 5
                for tidx, allies in enumerate(communities):
                    r = [len(a) for a in allies if len(a) > 1]
                    partitions = len(r)
                    if partitions >= top_a:
                        idx = tidx
                        break

                allies = communities[idx]
                allies = sorted(allies, key=lambda s: -len(s))[:top_a]
                random.shuffle(allies)

                alliesdf = pd.DataFrame(index=[f'Member {i + 1}' for i in range(max([len(a) for a in allies]))],
                                        columns=[f'Alliance {i + 1}' for i in range(len(allies))],
                                        data=''
                                        )
                for i, a in enumerate(allies):
                    at = list(a) + ([''] * (alliesdf.shape[0] - len(a)))
                    alliesdf[f'Alliance {i + 1}'] = at

                path = r"C:\school\PoliticalShapley\un_crawler\Allies"
                clear_folder(path, False)
                fpath = os.path.join(path, f'Allies GIRVAN {s_time} {e_time}.csv')
                alliesdf.to_csv(fpath, encoding='utf-8-sig', index=False)

            section_asyn_lpa = True
            if section_asyn_lpa:
                allies = list(community.asyn_lpa_communities(G, 'weight'))
                if len(allies) == 0:
                    print(f"Error for years: {s_time} {e_time}")
                    continue


                def get_coalition(powers, state):
                    for coal in powers:
                        c = [c.lower() for c in coal]
                        if state.lower() in c:
                            return coal
                    return None


                alliesdf = pd.DataFrame(index=[f'Member {i + 1}' for i in range(max([len(a) for a in allies]))],
                                        columns=[f'Alliance {i + 1}' for i in range(len(allies))],
                                        data=''
                                        )
                for i, a in enumerate(allies):
                    at = list(a) + ([''] * (alliesdf.shape[0] - len(a)))
                    alliesdf[f'Alliance {i + 1}'] = at

                path = r"C:\school\PoliticalShapley\un_crawler\Allies"
                clear_folder(path, False)
                fpath = os.path.join(path, f'Allies ASYN {s_time} {e_time}.csv')
                alliesdf.to_csv(fpath, encoding='utf-8-sig', index=False)

    section_analyze_migration = True
    if section_analyze_migration:
        path = r"C:\school\PoliticalShapley\un_crawler\Allies"
        g_paths = [os.path.join(path, t) for t in os.listdir(path) if 'GIRVAN' in t]
        a_paths = [os.path.join(path, t) for t in os.listdir(path) if 'ASYN' in t]
        dump_path = os.path.join(path, 'Over time')
        clear_folder(dump_path, delete_if_exist=True)

        visited = dict()
        q = queue.Queue()
        state = 'ISRAEL'
        q.put(state)
        visited[state] = True

        while not q.empty():
            state = q.get()
            print(f'[q: {q.qsize():>5}]\t{state}')

            for t_title, t_path in [('GIRVAN', g_paths), ('ASYN', a_paths)]:
                idf = pd.DataFrame(index=range(100))
                for gpath in a_paths:
                    m = re.search(r'([0-9]{4}) ([0-9]{4})', gpath)
                    s_time, e_time = int(m.group(1)), int(m.group(2))
                    time_sig = f'{s_time}-{e_time}'
                    # print(time_sig)
                    adf = pd.read_csv(gpath)

                    alliance = adf.eq(state).sum() > 0
                    if alliance.sum() > 0:
                        alliance_idx = alliance[alliance].index[0]
                        alliance = adf[alliance_idx]
                        alliance = alliance.sort_values().reset_index(drop=True)
                    else:
                        alliance = pd.Series(index=[0], data=[state])

                    idf.loc[alliance.index, time_sig] = alliance
                idf = idf[idf.isna().mean(axis=1) < 1]

                method_dump_path = os.path.join(dump_path, t_title)
                clear_folder(method_dump_path, delete_if_exist=False)
                df_path = os.path.join(method_dump_path, f'{state}.csv')
                idf.to_csv(df_path, encoding='utf-8-sig', index=False)

                for s in pd.unique(idf.values.ravel('K')):
                    if s in visited:
                        continue
                    else:
                        q.put(s)
                        visited[s] = True
