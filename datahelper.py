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
    nlp = spacy.blank('en')
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    nlp.to_disk("psy_ner")

def test_model(model, text):
    doc = nlp(text)
    results = []
    for ent in doc.ents:
        print (ent.text, ent.label_)
        entities.append((ent.start_char, ent.end_char, ent.label_))
    if len(entities) > 0:
        results = [text, {"entities": entities}]
    return results