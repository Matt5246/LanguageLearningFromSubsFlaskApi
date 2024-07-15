from flask import Flask, request, jsonify
import spacy
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

nlp = spacy.load('de_core_news_md')

model_name = 'Helsinki-NLP/opus-mt-de-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.route('/nlp', methods=['POST'])
def analyze_text():
    data = request.json
    if not data or 'word' not in data:
        return jsonify({'error': 'No word provided in the request body'}), 400

    word = data['word']

    doc = nlp(word)
    tokens_with_pos = {'lemma': doc[0].lemma_, 'pos': doc[0].pos_}
    print(word,tokens_with_pos)
    return jsonify({'result': tokens_with_pos})

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    texts = data["data"]

    if not texts:
        return jsonify({'error': 'Texts to translate are required'}), 400

    if isinstance(texts, str):
        texts = [texts]

    translated_texts = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")

        with tokenizer.as_target_tokenizer():
            outputs = model.generate(**inputs)

        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_texts.append(translated_text)

    return jsonify({'data': translated_texts})


@app.route('/')
def home():
    return 'Welcome to the Language Learning Flask API!'



if __name__ == '__main__':
    app.run(debug=True, port=8000)
