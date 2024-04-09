from flask import Flask, request, jsonify
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

@app.route('/nlp', methods=['POST'])
def analyze_text():
    data = request.json
    texts = data

    if not texts:
        return {'error': 'No texts provided in the request body'}, 400
    
    results = []
    for text in texts:
        doc = nlp(text)
        # Extract named entities
        entities = [ent.text for ent in doc.ents]
        # Extract tokens with lemma and POS tagging
        tokens_with_pos = [{'lemma': token.lemma_, 'pos': token.pos_} for token in doc]
        results.append({'entities': entities, 'tokens': tokens_with_pos})
    
    return {'results': results}

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    texts = data["data"]

    if not texts:
        return jsonify({'error': 'Texts to translate are required'}), 400

    if isinstance(texts, str):
        texts = [texts]

    # Translate each text individually
    translated_texts = []
    for text in texts:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt")

        # Perform translation
        with tokenizer.as_target_tokenizer():
            outputs = model.generate(**inputs)

        # Decode the translated output
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_texts.append(translated_text)

    return jsonify({'data': translated_texts})

@app.route('/translateStatic', methods=['GET'])
def translate_static():

    data = [
  'Lonelygirl15 war vor 11 Jahren die erste große\n' +
    'YouTuberin, die ähnlich wie Bibi, Dagi und',
  'Co mit ihren Zuschauern über Themen wie Eltern,\n' +
    'Schule und den ersten Kuss sprach!',
  'Eine halbe Millionen Klicks machte sie in\n' +
    'kurzer Zeit - das war damals ein Riesen-Erfolg,',
  'denn YouTube wurde da noch nicht so krass\ngenutzt wie heute.',
  'Doch irgendwann kam den Fans die YouTuberin\n' +
    'Bree komisch vor, denn für die damalige Zeit',
  'war die Musik zu perfekt abgemischt und die\n' +
    'Kameraführung wirkte oft zu professionell.',
  'Außerdem stellte ein Fan fest, dass die gesamte\n' +
    'Einrichtung von Brees Zimmer aus demselben',
  'Geschäft stammte.',
  'Und schon bald darauf flog der ganze Fake\nauf.',
  'Ein Journalist machte sich auf die Suche nach\n' +
    'LonelyGirl15 und schickte ihr auf MySpace',
  'einen versteckten Link, mit dem er ihren Standort\nrausfand.',
  'Und tatsächlich, die Nachricht wurde in einer\nSchauspielagentur geöffnet.',
  'Schnell wurde klar: LonelyGirl15 ist keine\n' +
    '15-jährige Amerikanerin, sondern eine 19',
  'jährige Schauspielerin aus Neuseeland, die\nvon der US-Firma gebucht wurde.',
  'Alles was Bree, die eigentlich Jessica heiß,\n' +
    'in ihren Videos erzählt hatte, war ihr nie',
  'passiert.',
  'Viele User deabonnierten stinksauer ihren\n' +
    'Kanal und ihre YouTube-Karriere war auf einen',
  'Schlag vorbei.',
  'Du willst wissen, wie Bree, die mittlerweile\n' +
    'verheiratet ist, heute aussieht und was sie',
  'inzwischen über ihren Fake-Kanal denkt?',
  'Du erfährst es im Link in der Infobox.'
]

    
    if not data:
        return jsonify({'error': 'Texts to translate are required'}), 400

    # Translate each text individually
    translated_texts = []
    for text in data:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt")

        # Perform translation
        with tokenizer.as_target_tokenizer():
            outputs = model.generate(**inputs)

        # Decode the translated output
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_texts.append(translated_text)

    return jsonify({'data': translated_texts})

@app.route('/')
def home():
    return 'Welcome to the Language Learning Flask API!'



if __name__ == '__main__':
    app.run(debug=True)
