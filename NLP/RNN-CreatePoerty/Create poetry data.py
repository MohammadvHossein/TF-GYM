import requests
from lxml import html

for i in range(1, 1995):
    page = requests.get('https://ganjoor.net/khayyam/robaee/sh' + str(i) + '/')
    tree = html.fromstring(page.content)
    with open('khayyam.txt', 'a', encoding='utf-8') as f:  # Specify UTF-8 encoding
        for t in tree.xpath('//div[@class="b"]/div/p/text()'):
            f.write('|' + str(t.replace('\u200c', ' ')) + '\n')