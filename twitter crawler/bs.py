import requests
from bs4 import BeautifulSoup

# Define the account you want to scrape tweets from
username = 'Gafni_Moshe'

# Scrape all tweets from the account
alltweets = []
url = 'https://twitter.com/' + username
while True:
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    tweets = soup.find_all('div', {'class': 'tweet'})
    for tweet in tweets:
        alltweets.append(tweet.find('p').text)
    next_button = soup.find('div', {'class': 'stream-container'}).find('div', {'class': 'stream-item-footer'}).find('a')
    if 'disabled' in next_button.attrs:
        break
    else:
        url = 'https://twitter.com' + next_button.attrs['href']

# Print all of the scraped tweets
for tweet in alltweets:
    print(tweet)
