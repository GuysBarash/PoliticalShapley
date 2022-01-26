import datetime
import os
import shutil
import re
import time
import json
from copy import deepcopy
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from IPython.display import display, HTML

from selenium import webdriver
from selenium.webdriver.chrome.options import Options, DesiredCapabilities
from selenium.webdriver.common.proxy import Proxy, ProxyType
from selenium.webdriver.support.ui import WebDriverWait

from tqdm import tqdm

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


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
        driver = webdriver.Chrome(r"C:\school\PoliticalShapley\Knesset_crawler\chromedriver.exe")
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
        return ret

    def __init__(self):
        self.root_path = os.path.dirname(__file__)
        self.outdir = os.path.join(self.root_path, 'data')
        self.json_dir = os.path.join(self.outdir, 'Resolutions jsons')
        self.reslinksdf_path = os.path.join(self.outdir, 'Links.csv')

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

    def investigate_all_resolutions(self, parallel=True):
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

        if parallel:
            pool = ThreadPool()
            res = pool.imap_unordered(func=UNcrawler.crawl_single_resolution, iterable=inputs)
            time.sleep(0.1)
            for idx in tqdm(range(len(inputs)), desc='Scanning Resolutions [Parallel]'):
                outp_pckg = res.next()
            time.sleep(0.1)
            pool.close()
            pool.join()
            time.sleep(0.1)
        else:
            for inp_pckg in tqdm(inputs, desc='Scanning Resolutions [Concurrent]'):
                outp_pckg = UNcrawler.crawl_single_resolution(inp_pckg)


if __name__ == '__main__':
    crawler = UNcrawler()
    crawler.search_records()
    crawler.investigate_all_resolutions()
