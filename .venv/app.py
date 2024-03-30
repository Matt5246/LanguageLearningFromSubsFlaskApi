from flask import Flask, request
import spacy
from transformers import MarianMTModel, MarianTokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the German language model
nlp = spacy.load('de_core_news_md')
german_text = "Hallo, wie geht es dir?"
model_name = 'Helsinki-NLP/opus-mt-de-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.route('/')
def analyze_text():
    text = "Das ist ein Beispieltext heute."
    doc = nlp(text)
    
    # Extract named entities
    entities = [ent.text for ent in doc.ents]
    
    # Extract tokens with lemma and POS tagging
    tokens_with_pos = [{'token': token.lemma_, 'pos': token.pos_} for token in doc]
    
    return {'entities': entities, 'tokens': tokens_with_pos}
@app.route('/translate', methods=['POST'])
def translate():
    # Get the text to be translated from the POST request
    text = request.json.get('text')
    if not text:
        return {'error': 'Text to translate is required'}, 400

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Perform translation
    with tokenizer.as_target_tokenizer():
        outputs = model.generate(**inputs)

    # Decode the translated output
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {'translated_text': translated_text}


if __name__ == '__main__':
    app.run(debug=True)
