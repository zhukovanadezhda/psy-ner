import json
import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
import os
import random
import requests
import warnings
import xmltodict
from unidecode import unidecode

def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def upgrade_data(file):
    objects = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            objects.append(line.lower().strip())
            objects.append(line.capitalize().strip())
            objects.append(line.upper().strip())
    for object in objects:
        with open(file, "a", encoding="utf-8") as f:
            f.write(f"{object} \n")

def delete_dublicates(file):
    with open(file, "r") as f:
        lines = f.readlines()
    lines = list(set(lines))
    with open(file, "w") as f:
        f.writelines(lines)

def save_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
def create_training_data(patterns, file, type):
    data = load_data(file)
    for item in data:
        pattern = {
                    "label": type,
                    "pattern": item
                    }
        patterns.append(pattern)
    return patterns

def generate_rules(patterns):
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    nlp.to_disk("psy_ner")

def test_model(model, text):
    text = unidecode(text)
    doc = model(text)
    results = []
    entities = []
    #with open(filename, 'a', encoding="utf-8") as f:
        #f.write(f'{text} \n\n')
    for ent in doc.ents:
            #f.write(f'{ent.text} {ent.label_} \n')
        entities.append((ent.start_char, ent.end_char, ent.label_))
        #f.write(f'\n')
    if len(entities) > 0:
        results = [text, {"entities": entities}]
    return results


def get_summary(PMID):
    """Obtaining information about an article published in PubMed using the PubMed API.
    Parameters
    ----------
    PMID : int
        The PubMed id of the article.
    access_token : str
        Access token for github.
    log_file : str
        A file to store information about the errors provided by this function.
    Returns
    -------
    summary : dictionary
        A dictionary obtained from xml format provided by pubmed api entrez.
        
    Example of query
    -------
    https://www.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=36540970&retmode=xml&rettype=abstract
    
    """
    db = 'pubmed'
    domain = 'https://www.ncbi.nlm.nih.gov/entrez/eutils'
    retmode = 'xml'
    queryLinkSearch = f'{domain}/efetch.fcgi?db={db}&id={PMID}&retmode{retmode}&rettype=abstract'  
    response = requests.get(queryLinkSearch)
    summary = xmltodict.parse(response.content)
    
    return summary


def get_abstract_from_summary(summary):
    """Obtaining abstract from the dictionary with summary returned by PubMed API.
    Parameters
    ----------
    summary : dictionary
        A dictionary obtained from xml format provided by pubmed api entrez.
    log_file : str
        A file to store information about the errors provided by this function.
    Returns
    -------
    abstract : str
        The article abctact.
        
    """
    
    try:
        article = summary['PubmedArticleSet']['PubmedArticle']
        abstract_raw = article['MedlineCitation']['Article']['Abstract']['AbstractText']
        if isinstance(abstract_raw, list):
            abstract = ""
            for d in abstract_raw:
                abstract += d['#text'] + " "    
        elif isinstance(abstract_raw, dict):
            abstract = ""
            abstract += abstract_raw['#text'] + " "
        else:
            abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText']  

        return abstract
    except:
        print('No abstract')
        return None