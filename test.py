import requests
from bs4 import BeautifulSoup

res = requests.get('https://news.ycombinator.com/news')
soup = BeautifulSoup(res.text, 'html.parser')
link = soup.select('.athing')
vote = soup.select('.score')

def create_custom(link, vote):
    hn = []
    for idx, item in enumerate(link):
        title = link[idx].getText()
        href = link[idx].get('href', None)
        hn.append({'title': title,
                   'link': href})
    return hn

print(create_custom(link=link, vote=vote))