import numpy as np
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from spacy.training import offsets_to_biluo_tags


def get_cleaned_label(label: str):
    """
    This function cleans a label by removing the tag prefix and returns the cleaned label.
    
    Parameters
    ----------
    label : str
        The label to be cleaned.
    Returns
    -------
    str
        The cleaned label.
        
    """
    if "-" in label:
        return label.split("-")[1]
    else:
        return label
    
    
def create_total_target_vector(nlp, docs):
    """
    This function creates a target vector for a set of documents. The target vector contains the expected labels for each
    entity in the documents.
    
    Parameters
    ----------
    nlp : spacy.lang object
        The spaCy language object to use for entity recognition.
    docs : list
        A list of documents, where each document is a tuple containing the text and a dictionary of entity annotations.
        
    Returns
    -------
    list
        A list of labels, where each label corresponds to an entity in the documents.
        
    """
    target_vector = []
    for doc in docs:
        #print (doc)
        new = nlp.make_doc(doc[0])
        entities = doc[1]["entities"]
        bilou_entities = offsets_to_biluo_tags(new, entities)
        final = []
        for item in bilou_entities:
            final.append(get_cleaned_label(item))
        target_vector.extend(final)
    return target_vector


def create_prediction_vector(nlp, text):
    """
    This function creates a prediction vector for a given text. The prediction vector contains the predicted labels for each
    entity in the text.
    
    Parameters
    ----------
    nlp : spacy.lang object
        The spaCy language object to use for entity recognition.
    text : str
        The text to be processed.
        
    Returns
    -------
    list
        A list of labels, where each label corresponds to an entity in the text.
        
    """
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions(nlp, text)]


def create_total_prediction_vector(nlp, docs: list):
    """
    This function creates a prediction vector for a set of documents. The prediction vector contains the predicted labels for each
    entity in the documents.
    
    Parameters
    ----------
    nlp : spacy.lang object
        The spaCy language object to use for entity recognition.
    docs : list
        A list of documents, where each document is a tuple containing the text and a dictionary of entity annotations.
    
    Returns
    -------
    list
        A list of labels, where each label corresponds to an entity in the documents.
        
    """
    prediction_vector = []
    for doc in docs:
        prediction_vector.extend(create_prediction_vector(nlp, doc[0]))
    return prediction_vector


def get_all_ner_predictions(nlp, text):
    """
    This function gets all the named entities and their labels in a given text using spaCy.
    
    Parameters
    ----------
    nlp : spacy.lang object
        The spaCy language object to use for entity recognition.
    text : str
        The text to be processed.
        
    Returns
    -------
    list
        A list of labels, where each label corresponds to a named entity in the text.
        
    """
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = offsets_to_biluo_tags(doc, entities)
    return bilou_entities


def get_model_labels():
    """
    Retrieve the named entity recognition (NER) labels used by the SpaCy model.
    
    Parameters
    ----------
    None

    Returns
    -------
    list
        A list of the NER labels used by the SpaCy model.
        
    """
    labels = list(nlp.get_pipe("ner").labels)
    labels.append("O")
    return sorted(labels)


def get_dataset_labels():
    """
    Retrieve the NER labels used in the dataset.
    
    Parameters
    ----------
    None

    Returns
    -------
    list
        A sorted list of the NER labels used in the dataset.
        
    """
    return sorted(set(create_total_target_vector(docs)))


def generate_confusion_matrix(nlp, docs): 
    """
    Generate a confusion matrix for the NER predictions of the SpaCy model.

    Parameters
    ----------
    nlp : object
        A SpaCy NLP object.
    docs : list
        A list of SpaCy Doc objects.

    Returns
    -------
    numpy.ndarray
        A confusion matrix of the NER predictions.
        
    """
    classes = sorted(set(create_total_target_vector(nlp, docs)))
    y_true = create_total_target_vector(nlp, docs)
    y_pred = create_total_prediction_vector(nlp, docs)
    #print (y_true)
    #print (y_pred)
    return confusion_matrix(y_true, y_pred, labels=classes)

def plot_confusion_matrix(nlp, docs, classes, normalize=False, cmap=pyplot.cm.Blues):
    """
    Plot a confusion matrix for the NER predictions of the SpaCy model.
    Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    nlp : object
        A SpaCy NLP object.
    docs : list
        A list of SpaCy Doc objects.
    classes : list
        A list of NER classes.
    normalize : bool, optional
        If True, normalize the confusion matrix. Default is False.
    cmap : colormap, optional
        The color map to use in the plot. Default is pyplot.cm.Blues.

    Returns
    -------
    numpy.ndarray
        A confusion matrix of the NER predictions.
        
    """
   
    title = 'Confusion Matrix, for SpaCy NER'

    # Compute confusion matrix
    cm = generate_confusion_matrix(nlp, docs)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    fig, ax = pyplot.subplots(figsize=(27,17))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                # Frame diagonal elements
                bbox_props = dict(boxstyle="round,pad=0.7", fc="white", ec="#1865AB", lw=2)
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                        bbox=bbox_props, color="black")
            else:
                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    pyplot.show()
    return cm