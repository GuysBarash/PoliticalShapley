import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options, DesiredCapabilities
from selenium.webdriver.common.proxy import Proxy, ProxyType

from bs4 import BeautifulSoup
import bs4
from selenium.webdriver.support.ui import WebDriverWait
from lxml import etree
from io import StringIO
import re
import os
import codecs
import shutil
import time
from tqdm import tqdm

import pandas as pd

if __name__ == '__main__':
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corpus')

    english_regex = '[a-zA-Z]+'
    hebrew_regex = '+[קראטוןםפשדגכעיחלךףזסבהנמצתץ]'


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


class KnessetCrawler:
    def __init__(self):
        self.base_url = r'https://m.knesset.gov.il/About/History/Pages/KnessetHistory.aspx?kns=XXX'
        self.work_path = os.path.dirname(__file__)
        self.raw_data = os.path.join(self.work_path, 'raw_knesset_stats')
        self.outputs_path = os.path.join(self.work_path, 'data')

        self.pages = None

        for i in range(1, 25):
            df = pd.DataFrame()
            path = os.path.join(r'C:\school\PoliticalShapley\Knesset_crawler\rawdfs', f'{i}.csv')
            df.to_csv(path, encoding='utf-8-sig')

        print("initialize driver")
        options = Options()
        # options.add_argument('--remote-debugging-port=9222')
        # options.add_experimental_option("useAutomationExtension", False)
        # options.set_capability("acceptInsecureCerts", True)
        # options.set_capability("acceptSslCerts", True)
        # options.add_experimental_option("excludeSwitches", ["enable-automation"])
        # options.add_experimental_option('excludeSwitches', ['enable-logging'])
        # options.add_experimental_option('useAutomationExtension', False)
        # options.add_argument('--disable-blink-features=AutomationControlled')
        self.driver = webdriver.Chrome()
        self.driver.wait = WebDriverWait(self.driver, 5)
        time.sleep(3)
        print("driver initialized")

    def index_sites(self):
        self.pages = [(os.path.join(self.raw_data, p), p) for p in os.listdir(self.raw_data) if '.html' in p]

    def extract(self):
        self.index_sites()
        for idx, (page, page_name) in enumerate(self.pages):
            try:
                # page = r'C:/school/PoliticalShapley/Knesset_crawler/raw_knesset_stats/2.html'
                self.driver.get(page)
                html = self.driver.page_source
                soup = bs4.BeautifulSoup(html, 'lxml')

                section_fetch_seats = True
                if section_fetch_seats:
                    pass

                section_fetch_date = False
                if section_fetch_date:
                    success = False
                    try:
                        a = soup.find_all('td', {'class': 'HistKnessetLinksTD'})[0]
                        t_soup = bs4.BeautifulSoup(a.decode_contents(), 'lxml')
                        st = t_soup.find_all('div')[0].text
                        term_start, term_end = re.findall(r'([0-9]+\.[0-9]+\.[0-9]+)', st)
                        success = True
                    except Exception as e:
                        pass

                    if not success:
                        try:
                            a = soup.find_all('td', {'class': 'HistKnessetLinksTD'})[0]
                            t_soup = bs4.BeautifulSoup(a.decode_contents(), 'lxml')
                            st = t_soup.find_all('strong')[0].text
                            term_start, term_end = re.findall(r'([0-9]+\.[0-9]+\.[0-9]+)', st)
                            success = True
                        except Exception as e:
                            pass

                    if not success:
                        try:
                            a = soup.find_all('div', {'class': 'Div4px'})[0]
                            t_soup = bs4.BeautifulSoup(a.decode_contents(), 'lxml')
                            st = t_soup.find_all('strong')[0].text
                            term_start, term_end = re.findall(r'([0-9]+\.[0-9]+\.[0-9]+)', st)
                            success = True
                        except Exception as e:
                            pass

                    if success:
                        print(f"({idx})({page_name})\t{term_start}\t-->\t{term_end}")
                    else:
                        raise Exception("Boing")

            except Exception as e:
                print(f"Failed on: {page}")

    def terminate(self):
        self.driver.quit()


if __name__ == '__main__':
    path = "C:\school\PoliticalShapley\Knesset_crawler\members\Book3.csv"
    rdf = pd.read_csv(path)
    datadf = None
    for cidx in rdf.columns:
        sr = pd.DataFrame(rdf[cidx])
        xcol = sr.columns[0]
        sr = sr[~sr[xcol].isna()]
        sr['isKnesset'] = sr[xcol].str.isnumeric()
        sr['isName'] = False
        sr.loc[0, 'isName'] = True

        tdf = pd.DataFrame(sr['isKnesset'].shift(1, fill_value=False))
        tdf['1'] = sr['isKnesset'].shift(1, fill_value=False)
        tdf['2'] = sr['isKnesset']
        tdf['X'] = ~tdf['1'] & ~tdf['2']
        sr['isName'] = tdf['X']

        sr['isParty'] = ~sr['isName'] & ~sr['isKnesset']
        sr['Party'] = np.NaN

        sr['Name'] = np.NaN
        sr.loc[sr['isName'], 'Name'] = sr.loc[sr['isName'], xcol]
        sr['Name'] = sr['Name'].fillna(method='ffill')

        sr.loc[sr['isParty'], 'Party'] = sr.loc[sr['isParty'], xcol]
        sr['Party'] = sr['Party'].shift(-1)
        sr = sr[~sr['isParty']]

        sr = sr.loc[sr['isKnesset']]
        sr['Knessent id'] = sr[xcol]
        sr = sr[['Name', 'Knessent id', 'Party']].sort_values(by=['Name', 'Knessent id'], ascending=False)

        if datadf is None:
            datadf = sr.copy()
        else:
            datadf = pd.concat([datadf, sr], ignore_index=True)

    datadf['Knessent id'] = datadf['Knessent id'].astype(int)
    datadf = datadf.sort_values(by=['Name', 'Knessent id'], ascending=False)

    datadf.to_csv(os.path.join(r'C:\school\PoliticalShapley\Knesset_crawler\ready data', 'MK.csv'),
                  encoding='utf-8-sig',
                  index=False,
                  )

    for idx in datadf['Knessent id'].unique():
        df = datadf[datadf['Knessent id'].eq(idx)]
        print(f'Knesset {idx} MKs: {len(df["Name"].unique())}')

if __name__ == '__main__' and False:
    pass
    # crwlr = KnessetCrawler()
    # crwlr.extract()
    # crwlr.terminate()

    root_path = os.path.dirname(__file__)
    dfs_path = os.path.join(root_path, 'rawdfs')

    dfs = [(os.path.join(dfs_path, tdf), tdf) for tdf in os.listdir(dfs_path) if '.csv' in tdf]
    datadf = None
    for dfidx, (df_path, df_name) in enumerate(dfs):
        knesset_idx = int(df_name.split('.')[0])
        df = pd.read_csv(df_path)
        df['knesset_id'] = knesset_idx

        if datadf is None:
            datadf = df.copy()
        else:
            datadf = pd.concat([datadf, df], ignore_index=True)
        print(f"File: {df_name}\t Size: {df.shape[0]}")

    df_cols_translate = dict()
    df_cols_translate['שם הרשימה'] = "Party"
    df_cols_translate["מספר קולות כשרים"] = "Valid votes"
    df_cols_translate['קולות באחוזים'] = "Votes Percentage"
    df_cols_translate['מספר מנדטים'] = "Seats"
    datadf = datadf.rename(columns=df_cols_translate)
    datadf["Votes Percentage"] /= 100.0

    datadf = datadf.sort_values(by=['Seats'], ascending=False)
    datadf['Token'] = ' '

    tokens = dict()
    section_define_tokens = True
    if section_define_tokens:
        tokens['עבודה'] = 'Avoda'
        tokens['העבודה'] = 'Avoda'
        tokens['מפא"י'] = 'Avoda'
        tokens['מערך'] = 'Avoda'
        tokens['המערך'] = 'Avoda'
        tokens['מפלגת העבודה בראשות מרב מיכאלי'] = 'Avoda'
        tokens['מפלגת העבודה בראשות אבי גבאי'] = 'Avoda'

        tokens['ליכוד'] = 'Likud'
        tokens['הליכוד'] = 'Likud'
        tokens['הליכוד בהנהגת בנימין נתניהו לראשות הממשלה'] = 'Likud'
        tokens['גח"ל'] = 'Likud'

        tokens['כחול לבן'] = 'Blue And White'
        tokens['כחול לבן בראשות בני גנץ ויאיר לפיד'] = 'Blue And White'
        tokens['כחול לבן בראשות בני גנץ'] = 'Blue And White'

        tokens['כולנו'] = 'Kulanu'
        tokens['כולנו בראשות משה כחלון'] = 'Kulanu'
        tokens['כולנו הימין השפוי בראשות משה כחלון'] = 'Kulanu'

        tokens['יש עתיד בראשות יאיר לפיד'] = 'Yesh Atid'
        tokens['יש עתיד'] = 'Yesh Atid'

        tokens['חד"ש'] = 'Israeli Communist Party'
        tokens['רק"י'] = 'Israeli Communist Party'
        tokens['מק"י'] = 'Israeli Communist Party'

        tokens['ש"ס'] = 'Shas'
        tokens['התאחדות הספרדים שומרי תורה - תנועת ש"ס'] = 'Shas'
        tokens['התאחדות הספרדים שומרי תורה תנועתו של מרן הרב עובדיה יוסף זצ"ל'] = 'Shas'
        tokens['התאחדות הספרדים שומרי התורה תנועתו של מרן הרב עובדיה יוסף זצ"ל'] = 'Shas'
        tokens['התאחדות הספרדים שומרי תורה תנועתו של מרן הרב עובדיה'] = 'Shas'

        tokens['מפד"ל'] = 'Religious Zionists'
        tokens['המפד"ל'] = 'Religious Zionists'
        tokens['הבית היהודי'] = 'Religious Zionists'
        tokens["הציונות הדתית בראשות בצלאל סמוטריץ'"] = 'Religious Zionists'
        tokens['הבית היהודי בראשות נפתלי בנט'] = 'Religious Zionists'
        tokens['האיחוד הלאומי-מפד"ל'] = 'Religious Zionists'

        tokens['ימינה בראשות נפתלי בנט'] = 'Yamina'
        tokens['ימינה'] = 'Yamina'

        tokens['מר"צ'] = 'Meretz'
        tokens['מר"ץ'] = 'Meretz'
        tokens['מרצ'] = 'Meretz'
        tokens['מרץ'] = 'Meretz'
        tokens['התנועה החדשה מר"צ'] = 'Meretz'

        tokens['ישראל ביתנו בראשות אביגדור ליברמן'] = 'Israel Beitenu'
        tokens['ישראל ביתנו'] = 'Israel Beitenu'

        tokens['אגו"י'] = 'Agudat Israel'

        tokens['שינוי-התנועה החילונית'] = 'Change'
        tokens['שינוי'] = 'Change'

    datadf = datadf.sort_values(by=['knesset_id', 'Seats'], ascending=False)
    for k, token in tokens.items():
        datadf.loc[datadf['Party'].eq(k), 'Token'] = token
    print(f"Entries: {datadf.shape[0]}")
    datadf.to_csv(os.path.join(root_path, 'Knesset.csv'), encoding='utf-8-sig')

    p = datadf.loc[datadf['Token'].eq(' '), 'Party'].value_counts()
    s = 'קדימה'
    for pt in [pt for pt in p.index if s in pt]:
        print(pt)
    print("\n")
    p = p[p > 1]
    print(p)
