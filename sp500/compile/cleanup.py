import json
import pandas as pd


headers = {'User-Agent': 'Mozilla/5.0',
           'Referer': "http://finviz.com/quote.ashx?t="}

url = headers['Referer'] + "AAPL" ##AAPL change into any user input ticker
response = requests.get(url, headers=headers)
root = lxml.html.fromstring(response.text)
news_rows = root.xpath("//table[@id='news-table']/tr")



