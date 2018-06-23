
"""
preprocess function to filter/prepare Wikipedia docs.
"""

"""
Wikipedia's dataset contains only one table called documents,
This table consists of two columns (address, text)and contains more than five millions raw
address is an address of text(document)
text is a document.
"""


import regex as re
from html.parser import HTMLParser
"""
HTMLParser(): parses a web page’s HTML/XHTML content and provides the information we are looking for.
"""
#https://www.pythoncentral.io/html-parser/

PARSER = HTMLParser()

# store the id of disambig pages that aren't catched by WikiExtractor in BLACKLIST.
"""
    WikiExtractor.py is a Python script that extracts and cleans text from a Wikipedia database dump.
"""
BLACKLIST = set(['23443579', '52643645'])

#preprocessing function: extract the rest disambig pages
def preprocess(article):

    # Take out HTML escaping WikiExtractor didn't clean
    """
        first, we will scan all rest articles, use  html.unescape(v) (which v is the context of article)
        to Convert all named and numeric character references (e.g. &gt;, &#62;, &x3e;)
        in the string v to the corresponding unicode characters.
    """
    # https://docs.python.org/3/library/html.html
    """
    article is dictionary contains k is a key and v is a value contains article context
    """
    for k, v in article.items():
        article[k] = PARSER.unescape(v)

    # Filter some disambiguation pages not caught by the WikiExtractor
    """
         with the WikiExtractor, the dump was processed and filtered for internal disambiguation,
         list, index, and outline pages(pages that are typically just links).
    """

    if article['id'] in BLACKLIST:
        return None

    if '(disambiguation)' in article['title'].lower():
        return None
    if '(disambiguation page)' in article['title'].lower():
        return None

    # Take out List/Index/Outline pages (mostly links)
    if re.match(r'(List of .+)|(Index of .+)|(Outline of .+)',
                article['title']):
        return None

    # Return doc with `id` set to `title`
    return {'id': article['title'], 'text': article['text']}
