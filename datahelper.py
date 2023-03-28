# default libraries
import json
import os
import random
import warnings

# installed libraries
import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import requests
import xmltodict
from unidecode import unidecode


def load_data(file):
    """
    Loads data from a JSON file and returns it as a dictionary.
    
    Parameters
    ----------
    file : str
        The path of the json file.

    Returns
    -------
    data : dict
    The content of the json file.
    
    """
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def upgrade_data(file):
    """
    Upgrades the data in a text file by adding lowercase, capitalized and uppercase variations of each line.
    
    Parameters
    ----------
    file : str
        The path of the file.

    Returns
    -------
    None
    
    """
    objects = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            # add lowercase, capitalized, and uppercase versions of the object
            objects.append(line.lower().strip())
            objects.append(line.capitalize().strip())
            objects.append(line.upper().strip())
    for object in objects:
        with open(file, "a", encoding="utf-8") as f:
            f.write(f"{object} \n")

def delete_dublicates(file):
    """
    Deletes duplicate lines from a text file.
    
    Parameters
    ----------
    file : str
        The path of the file.

    Returns
    -------
    None
    
    """
    with open(file, "r") as f:
        lines = f.readlines()
    # use a set to remove duplicates
    lines = list(set(lines))
    with open(file, "w") as f:
        f.writelines(lines)

def save_data(file, data):
    """
    Saves a dictionary as a JSON file.
    
    Parameters
    ----------
    file : str
        The path of the output json file.
    data : dict
        The dictionary to be saved.

    Returns
    -------
    None
    
    """
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
def create_training_data(patterns, file, type):
    """
    Creates training data for the entity ruler.
    
    Parameters
    ----------
    patterns : list
        The list of patterns.
    file : str
        The path of the file containing a list of strings.
    type : str
        The type of entities that the patterns belong to.

    Returns
    -------
    patterns : list
        The updated list of patterns.
    
    """
    data = load_data(file)
    for item in data:
        pattern = {
                    "label": type,
                    "pattern": item
                    }
        patterns.append(pattern)
    return patterns

def generate_rules(patterns):
    """ 
    Generates the entity ruler rules and save them to disk.
    
    Parameters
    ----------
    patterns : list
        The list of patterns.

    Returns
    -------
    None
    
    """
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    nlp.to_disk("annotation_ner")

def test_model(model, text):
    """
    Tests the entity recognition model on a given text. 
    
    Parameters
    ----------
    model : spacy language model
        The trained spacy language model.
    text : str
        The text to be processed.

    Returns
    -------
    results : list
        The list containing the original text and its annotated entities in the format [(start_char, end_char, label), ...]
    
    """
    # remove accents from the text
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

def pretty_colors():
    """
    Sets colors to the spacy function displacy do display found entities.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    options : dic
        A dictionary with different colors for each label.
    """
    colors = {
        "ANXIETY DISORDERS": "#F6A5C0",
        "BIPOLAR DISORDERS": "#F7D6CB",
        "DEPRESSIVE DISORDERS": "#C7CEEA",
        "DISSOCIATIVE DISORDERS": "#F7DDAA",
        "PSYCHEDELIC DRUGS": "#D3BCC0",
        "EATING DISORDERS": "#F7E5C8",
        "NEURO-COGNITIVE DISORDERS": "#A8D8EA",
        "NEURO-DEVELOPMENTAL DISORDERS": "#D5A6BD",
        "NON-SUBSTANCE RELATED DISORDERS": "#F4C9A4",
        "OBSESSIVE-COMPULSIVE AND RELATED DISORDERS": "#F1D1D0",
        "OTHER DISORDERS": "#C8BFE7",
        "PARAPHILIAS": "#D2E0E3",
        "PERSONALITY DISORDERS": "#F6B7B8",
        "SCHIZOPHRENIA SPECTRUM AND OTHER PSYCHOTIC DISORDERS": "#A2B7DC",
        "SEXUAL DYSFUNCTIONS": "#E6A7CF",
        "SLEEP-WAKE DISORDERS": "#F4B4C1",
        "SOMATIC SYMPTOM RELATED DISORDERS": "#B4D4E7",
        "SUBSTANCE-RELATED DISORDERS": "#F5C5D1",
        "TRAUMA AND STRESS RELATED DISORDERS": "#D1D2C9",
        "ELIMINATION DISORDERS": "#E8A9C7",
        "DISRUPTIVE IMPULSE-CONTROL, AND CONDUCT DISORDERS": "#D3E3F5",
        "SYMPTOMS": "#C3B7B7"
    }

    options = {"ents": [
        "ANXIETY DISORDERS",
        "BIPOLAR DISORDERS",
        "DEPRESSIVE DISORDERS",
        "DISSOCIATIVE DISORDERS",
        "PSYCHEDELIC DRUGS",
        "EATING DISORDERS",
        "NEURO-COGNITIVE DISORDERS",
        "NEURO-DEVELOPMENTAL DISORDERS",
        "NON-SUBSTANCE RELATED DISORDERS",
        "OBSESSIVE-COMPULSIVE AND RELATED DISORDERS",
        "OTHER DISORDERS",
        "PARAPHILIAS",
        "PERSONALITY DISORDERS",
        "SCHIZOPHRENIA SPECTRUM AND OTHER PSYCHOTIC DISORDERS",
        "SEXUAL DYSFUNCTIONS",
        "SLEEP-WAKE DISORDERS",
        "SOMATIC SYMPTOM RELATED DISORDERS",
        "SUBSTANCE-RELATED DISORDERS",
        "TRAUMA AND STRESS RELATED DISORDERS",
        "ELIMINATION DISORDERS",
        "DISRUPTIVE IMPULSE-CONTROL, AND CONDUCT DISORDERS",
        "SYMPTOMS"
    ],
        "colors": colors
    }
    
    return options

