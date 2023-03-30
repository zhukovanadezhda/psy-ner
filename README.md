# PsyNer : Custom Named Entity Recognition model for mental health purposes

## Introduction 

The World Health Organization (WHO) reports that mental health disorders affect approximately one in four individuals globally, with depression being a leading cause of disability. [1] However, nearly two-thirds of individuals with known mental disorders do not seek help from a professional. [2] The European region has the highest prevalence of mental disorders globally, with an estimated 164 million individuals affected, according to the WHO. 

Although conventional treatments such as antidepressant medications and talk therapy can be effective for some individuals, many of them do not respond well to these methods. Therefore, there is an urgent need for new and effective treatments for mental health disorders. Psychedelics have received much scientific attention in recent years due to their potential to help with mental health disorders. Research has shown that psychedelic drugs such as psilocybin, LSD, and MDMA have potential therapeutic benefits for mental health conditions such as depression, anxiety, and PTSD. A study published in The Lancet Psychiatry in 2021 found that psilocybin therapy was more effective than conventional antidepressant medication in treating depression. [3]

However, several challenges arise when researching mental health and psychedelics, which may limit productivity and delay much-needed relief. These challenges include the large volume of data on mental health disorders and the therapeutic use of psychedelics, inconsistent terminology used across different studies and sources, and a growing trend of analyzing social media [4] to understand people's experiences using psychedelics to treat mental health disorders. In some countries, studies with psychedelics are not allowed, making it difficult for researchers to study their therapeutic use. However, people may still use psychedelics illegally and share their experiences on social media platforms. Analyzing this data may provide some insights into the potential therapeutic benefits of psychedelics for mental health disorders. In fact, some studies have already analyzed social media data to explore people's experiences with psychedelic use for mental health conditions. [7]

An implementation of an appropriate technical solution would assiste in resolving these issues and provide a powerfull tool to strengthen this branch of studies. 

Named Entity Recognition (NER) is a subfield of natural language processing (NLP) that aims to identify and extract named entities from unstructured text data. Named entities include specific types of objects, such as people, organizations, locations, dates, and other types of things that can be named. NER can be a challenging task as named entities can be expressed in many different ways and can have complex relationships with other entities and the surrounding text.

A customized NER model that can recognize mental disorders, psychedelic drugs, and related vocabulary has the potential to address some of the key challenges in mental health treatment. Such a model could help researchers quickly sift through the overwhelming amount of information [6] and identify relevant studies or data points, standardize the terminology, and ensure that all relevant studies and data are included in the analysis. Moreover, it could help monitor discussions about mental health and psychedelics use on social media and identify potential areas for further study.

Thus, a custom NER model can improve the accuracy of mental health diagnoses [5], support research and development efforts, and reinforce the study of the therapeutic use of psychedelic drugs. The model's importance lies in its ability to identify relevant data more efficiently, enabling professionals to make informed decisions and take appropriate action.

[1] https://www.who.int/news-room/fact-sheets/detail/mental-disorders

[2] https://www.who.int/news/item/28-09-2001-the-world-health-report-2001-mental-disorders-affect-one-in-four-people

[3] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7643046/

[4] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6069295/

[5] https://www.frontiersin.org/articles/10.3389/fdgth.2022.1065581/full

[6] https://www.researchgate.net/profile/Holger-Froehlich-4/publication/358779855_Deep_Learning-based_Detection_of_Psychiatric_Attributes_from_German_Mental_Health_Records/links/62f13b870b37cc34477e99df/Deep-Learning-based-Detection-of-Psychiatric-Attributes-from-German-Mental-Health-Records.pdf?origin=publication_detail

[7] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10012542/

## Implementatioin

To create a custom NER model, three major steps were followed:


#### 1. Data collection and annotation

Abstracts of articles containing the words "psychedelics", "anxiety", and "depression" were collected using the PubMed API Entrez. These abstracts were then annotated using a rule-based NER model created specifically for this purpose. The annotated data was stored in the .spacy format for subsequent use. The data_txt folder contains the list of words for each label, the data_json folder contains the same files in .json format, and the data folder contains the final version of the training and test data in both .json (for spaCy v2) and .spacy (for spaCy v3) formats. The annotation_ner folder contains the rule-based NER model used to annotate the abstracts, and the notebook annotation.ipynb shows all the annotation steps.

#### 2. Training of the model

A base_config.cfg file was created using https://spacy.io/usage/training#config, which was then converted to the config.file for training the model. The model was trained on 334 texts and tested on 175 texts. The model (both last and best) is stored in the model folder. The notebook psyner.ipynb shows all the training steps, and the datahelper.py script contains all necessary functions.

#### 3. Evaluation of the model's performance

\* commimg soon *

