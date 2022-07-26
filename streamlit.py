import streamlit as st
import pandas as pd
import numpy
import requests
from bs4 import BeautifulSoup
import os
from medcat.cat import CAT
import re
from wikiapi import WikiApi

st.title('ExplainHealth')

# st.write("Here's our first attempt at using data to create a table:")
# st.write(pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# }))

# def load_data(url):
#     # url = 'http://eyerounds.org/cases.htm'
#     resp = requests.get(url)
#     soup = BeautifulSoup(resp.text, 'html.parser')
#     class_list = soup.find_all(class_ = 'col-xs-12 col-sm-6 col-md-6 col-lg-4')
#     link_list = []
#
#     for i in range(len(class_list)):
#         class_a = class_list[i].find_all('a')
#         for link in class_a:
#             temp = link.get('href')
#             start = temp.startswith('http://webeye.ophth.uiowa.edu/eyeforum/')
#             if start == False:
#                 temp = 'http://webeye.ophth.uiowa.edu/eyeforum/'+temp
# #         print(temp)
#             if temp.count('/')==5:
#                 link_list.append(temp)
#                 # print(temp)
#             else: continue
#
#     return link_list

def method(path):
    file_open = open(path, 'r')
    f = file_open.read()
    file_open.close()
    return f

def medterm_identify(text):
    cat = CAT.load_model_pack('medmen_wstatus_2021_oct')
    entities = cat.get_entities(text)

    med_terms = []
    for key in entities['entities']:
        result = entities['entities'][key]['pretty_name']
        result = result.lower()
        med_terms.append(result)

    med_terms_unique = list(dict.fromkeys(med_terms))

    return med_terms_unique

def translation(terms):
    wiki = WikiApi()
    wiki = WikiApi({ 'locale' : 'en'})
    # results = wiki.find(terms)
    # article = wiki.get_article(results[0])
    # summary = article.summary

    search = re.compile(r"^([^.]*).*")

    wiki_dict = {}
    for i in range(len(terms)):
        results = wiki.find(terms[i])
        length = len(results)

        if len(results) == 0:
            continue

        else:
            article = wiki.get_article(results[0])
            summary = article.summary
            m = re.match(search, summary)
#         wiki_dict[med_terms[i]] = m.group(1)
            wiki_first = m.group(1)
            wiki_dict[terms[i]] = wiki_first

    return wiki_dict

def output(text, dict):
    for key, value in dict.items():
        combin = key + '(' + value + ')'
        text = text.replace(key, combin)
    return text

path = 'dataset/12_Neurology.txt'
# file = method(path)

file = st.text_input('records')

st.write(file)
med_terms = medterm_identify(file)
wiki_dict = translation(med_terms)
output = output(file, wiki_dict)

if st.button('translate'):
    output
