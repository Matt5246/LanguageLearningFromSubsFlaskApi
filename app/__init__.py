from flask import Flask, request, jsonify
import spacy
from transformers import MarianMTModel, MarianTokenizer


app = Flask(__name__)

nlp_de = spacy.load('de_core_news_md')
nlp_ja = spacy.load('ja_core_news_md')
nlp_en = spacy.load('en_core_web_md')
nlp_pl = spacy.load('pl_core_news_md')
nlp_zh = spacy.load('zh_core_web_md')

model_name = 'Helsinki-NLP/opus-mt-de-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

@app.route('/nlp', methods=['POST'])
def analyze_text():
    data = request.json
    if not data or 'word' not in data:
        return jsonify({'error': 'No word provided in the request body'}), 400

    word = data['word']
    sourceLang = data['sourceLang']
    doc = None
    if not sourceLang:
        return jsonify({'error': 'Source language is required'}), 400
    if sourceLang == 'de':
        doc = nlp_de(word)
    elif sourceLang == 'ja':
        doc = nlp_ja(word)
    elif sourceLang == 'en':
        doc = nlp_en(word)
    elif sourceLang == 'pl':
        doc = nlp_pl(word)
    elif sourceLang == 'zh':
        doc = nlp_zh(word)
    elif sourceLang == 'auto':
        doc = nlp_de(word)
    else:
        return jsonify({'error': f"Source language is not currently used {sourceLang}"}), 400
    
    tokens_with_pos = {'lemma': doc[0].lemma_, 'pos': doc[0].pos_}
    print(word,tokens_with_pos)
    return jsonify({'result': tokens_with_pos})


@app.route('/frequency', methods=['POST'])
def frequencyCheck():
    try:
        data = request.json
        if not data or 'uniqueWords' not in data:
            return jsonify({'error': 'Invalid input'}), 400

        unique_words = data['uniqueWords']
        lemmatized_words = []

        for word in unique_words:
            cleaned_word = word.strip().lower()
            doc = nlp(cleaned_word)

            if len(doc) > 0:
                lemma = doc[0].lemma_.strip().lower()
                if lemma != cleaned_word:
                    lemmatized_words.append({'word': cleaned_word, 'lemma': lemma})
                else:
                    lemmatized_words.append({'word': cleaned_word, 'lemma': cleaned_word})
            else:
                lemmatized_words.append({'word': cleaned_word, 'lemma': cleaned_word})

        return jsonify({'data': lemmatized_words})
    except Exception as e:
        print(f"Error processing word frequency: {e}")
        return jsonify({'error': str(e)}), 500



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
    app.run(debug=False, port=8080)
