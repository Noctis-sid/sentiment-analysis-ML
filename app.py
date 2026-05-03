"""
CT052-3-M-NLP | Sentiment Analysis Web App
Flask backend — serves model predictions via REST API

Run:  python app.py
Open: http://127.0.0.1:5000
"""

import re
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load saved model ─────────────────────────────────────────
model = joblib.load('best_model.pkl')
print("✅ Model loaded successfully")

# ── Preprocessing (same as notebook) ─────────────────────────
emoticon_dict = {
    ':)':'happy', ':-)':'happy', ':D':'very_happy', ':-D':'very_happy',
    ':(':  'sad',  ':-(':'sad',  ';)':'wink',    ':P':'playful',
    ':-P':'playful', ':/' :'skeptical', '>:(':'angry',
    ':|':'neutral', '<3':'love', 'XD':'laughing', ':o':'surprised',
}
slang_dict = {
    'omg':'oh my god','lol':'laughing out loud','tbh':'to be honest',
    'ngl':'not going to lie','imo':'in my opinion','fav':'favourite',
    'luv':'love','gr8':'great','b4':'before','cuz':'because'
}
num_map = {str(i): w for i, w in enumerate(
    ['zero','one','two','three','four','five','six','seven','eight','nine'])}

def replace_emoticons(text):
    for e, m in emoticon_dict.items():
        text = text.replace(e, f' {m} ')
    return text

def replace_slangs(text):
    return ' '.join([slang_dict.get(w, w) for w in text.split()])

def handle_negation(text):
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcan't\b", "cannot",   text, flags=re.IGNORECASE)
    text = re.sub(r"n't\b", " not", text)
    return re.sub(r"\b(not|no|never|neither|nor|hardly|barely|scarcely)\s+(\w+)",
                  lambda m: m.group(1)+'_'+m.group(2), text)

def handle_contrast(text):
    pattern = r'\b(but|however|yet|although|though|despite|nevertheless|still|actually)\b'
    parts   = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) >= 3:
        text = parts[0]+' '+parts[1]+' '+(parts[2]+' ')*3
    return text

def clean_text(text):
    text = replace_emoticons(text)
    text = replace_slangs(text)
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d', lambda m: num_map[m.group(0)], text)
    text = handle_contrast(text)
    text = handle_negation(text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[^a-z_\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json()
    review = data.get('review', '').strip()

    if not review:
        return jsonify({'error': 'Empty review'}), 400

    cleaned = clean_text(review)
    pred    = model.predict([cleaned])[0]
    proba   = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)

    confidence = {cls: round(float(prob)*100, 1)
                  for cls, prob in zip(classes, proba)}

    return jsonify({
        'prediction' : pred,
        'confidence' : confidence,
        'is_positive': pred == 'Good'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)