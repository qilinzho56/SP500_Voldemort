import requests
import lxml.html


headers = {'User-Agent': 'Mozilla/5.0',
           'Referer': "http://finviz.com/quote.ashx?t="}

url = headers['Referer'] + "AAPL" ##AAPL change into any user input ticker
response = requests.get(url, headers=headers)
root = lxml.html.fromstring(response.text)
news_rows = root.xpath("//table[@id='news-table']/tr")
days_visited = 0
max = 2

for row in news_rows:
    if row.xpath("./td[contains(text(), 'Jan')]"):
        days_visited += 1
    if days_visited > max:
        break
    headline = row.xpath(".//a[@target='_blank']/text()")
    print(headline)