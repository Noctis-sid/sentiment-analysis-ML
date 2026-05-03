# ============================================================
# CT052-3-M-NLP | BONUS — Aspect-Based Sentiment Analysis
#
# What is ABSA?
#   Instead of giving ONE sentiment for the whole review,
#   ABSA finds WHICH ASPECTS are mentioned (story, acting,
#   music, action) and gives SEPARATE sentiment for each.
#
#   Example:
#   "The story was brilliant but the action was terrible"
#       └── story  → ✅ GOOD
#       └── action → ❌ BAD
#
# Reference: Slides 39–42 (CT052-3-M-NLP Week 8)
#            Blair-Goldensohn et al. (2008)
#            Hu & Liu (2004) — KDD
#
# Approach (as described in lecture slides):
#   1. Split review into clauses on contrast/comma markers
#   2. Match each clause to a known movie aspect
#   3. Run trained LR model on each clause
#   4. Output per-aspect sentiment summary
# ============================================================

import re
import joblib
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD TRAINED LR MODEL
# The LR model was trained and saved by the main script.
# It is a Pipeline (TF-IDF + LR) so it handles text directly.
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("BONUS: Aspect-Based Sentiment Analysis (ABSA)")
print("=" * 60)

try:
    model = joblib.load('lr_sentiment_model.pkl')
    print("  ✅ Trained LR model loaded: lr_sentiment_model.pkl")
except FileNotFoundError:
    print("  ❌ ERROR: lr_sentiment_model.pkl not found!")
    print("     Please run logistic_regression_sentiment.py first")
    print("     to train and save the model.")
    exit()

# ─────────────────────────────────────────────────────────────
# STEP 2 — DEFINE MOVIE ASPECTS + KEYWORDS
#
# Each aspect has a list of trigger words.
# If a clause contains any of these words, it belongs to
# that aspect. This is the keyword-matching approach described
# in Hu & Liu (2004) and your lecture slides (Slide 40).
# ─────────────────────────────────────────────────────────────
MOVIE_ASPECTS = {
    'Story / Plot': [
        'story', 'plot', 'script', 'writing', 'narrative',
        'screenplay', 'ending', 'twist', 'storyline', 'premise',
        'beginning', 'climax', 'pacing', 'pace'
    ],
    'Acting': [
        'acting', 'actor', 'actress', 'cast', 'performance',
        'character', 'role', 'played', 'portray', 'lead',
        'supporting', 'dialogue', 'delivery', 'emotion'
    ],
    'Direction': [
        'director', 'direction', 'directed', 'cinematography',
        'camera', 'shot', 'scene', 'visual', 'editing',
        'cut', 'lighting', 'framing', 'composition'
    ],
    'Action / Effects': [
        'action', 'effects', 'cgi', 'special effects',
        'stunt', 'fight', 'explosion', 'vfx', 'animation', 'scene',
        'graphics', 'sequence', 'choreography', 'battle', 'action scene'
    ],
    'Music / Sound': [
        'music', 'soundtrack', 'score', 'song', 'audio',
        'sound', 'background music', 'composer', 'theme',
        'melody', 'noise', 'mixing', 'bgm'
    ],
    'Emotions / Feel': [
        'emotional', 'boring', 'exciting', 'funny', 'scary',
        'touching', 'moving', 'thrilling', 'hilarious',
        'dull', 'slow', 'fast', 'intense', 'dramatic',
        'heartwarming', 'tear', 'laugh', 'cry', 'feel'
    ]
}

# ─────────────────────────────────────────────────────────────
# STEP 3 — PREPROCESSING (same as LR model)
# ─────────────────────────────────────────────────────────────
def handle_negation(text):
    text = re.sub(r"n't\b", " not", text)
    negations = r"\b(not|no|never|neither|nor|hardly|barely|scarcely)\s+(\w+)"
    return re.sub(negations, lambda m: m.group(1) + '_' + m.group(2), text)

def handle_contrast(text):
    contrast_pattern = r'\b(but|however|yet|although|though|despite|nevertheless|still|actually|in fact|turned out)\b'
    parts = re.split(contrast_pattern, text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) >= 3:
        text = parts[0] + ' ' + parts[1] + ' ' + (parts[2] + ' ') * 3
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = handle_contrast(text)
    text = handle_negation(text)
    text = re.sub(r'[^a-z_\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─────────────────────────────────────────────────────────────
# STEP 4 — CLAUSE SPLITTER
#
# Splits a review into smaller clauses at:
#   - Contrast words : but, however, although, though, yet
#   - Punctuation    : , ; .
# Each clause is then analysed separately.
# ─────────────────────────────────────────────────────────────
def split_into_clauses(review):
    """Split review into clauses at contrast words and punctuation."""
    # Split on contrast words and punctuation
    split_pattern = r'\b(but|however|although|though|yet|while|whereas|nevertheless|despite|still)\b|[,;.]'
    raw_clauses = re.split(split_pattern, review, flags=re.IGNORECASE)

    clauses = []
    for c in raw_clauses:
        if c is None:
            continue
        c = c.strip()
        # Skip very short fragments and the contrast words themselves
        if len(c.split()) >= 3 and c.lower() not in [
            'but', 'however', 'although', 'though', 'yet',
            'while', 'whereas', 'nevertheless', 'despite', 'still'
        ]:
            clauses.append(c)

    return clauses if clauses else [review]

# ─────────────────────────────────────────────────────────────
# STEP 5 — ASPECT MATCHER
#
# Checks which aspect a clause belongs to based on keywords.
# If no aspect keyword found → labelled as 'General'
# ─────────────────────────────────────────────────────────────
def match_aspect(clause):
    """Returns the aspect name that best matches the clause."""
    clause_lower = clause.lower()
    for aspect, keywords in MOVIE_ASPECTS.items():
        for kw in keywords:
            if kw in clause_lower:
                return aspect
    return 'General'

# ─────────────────────────────────────────────────────────────
# STEP 6 — MAIN ABSA FUNCTION
# ─────────────────────────────────────────────────────────────
def absa_predict(review):
    """
    Full ABSA pipeline:
    1. Split review into clauses
    2. Match each clause to an aspect
    3. Predict sentiment for each clause using LR model
    4. Return structured results
    """
    clauses = split_into_clauses(review)
    results = []

    for clause in clauses:
        aspect    = match_aspect(clause)
        cleaned   = clean_text(clause)
        sentiment = model.predict([cleaned])[0]
        proba     = model.predict_proba([cleaned])[0]
        confidence= max(proba) * 100

        results.append({
            'clause'    : clause,
            'aspect'    : aspect,
            'sentiment' : sentiment,
            'confidence': confidence
        })

    return results

def print_absa_results(review, results):
    """Pretty print ABSA results."""
    print(f"\n{'=' * 60}")
    print(f"  REVIEW: \"{review[:100]}{'...' if len(review) > 100 else ''}\"")
    print(f"{'=' * 60}")
    print(f"  {'ASPECT':<22} {'SENTIMENT':<10} {'CONFIDENCE':<12} CLAUSE")
    print(f"  {'-' * 57}")

    for r in results:
        icon = '✅' if r['sentiment'] == 'Good' else '❌'
        aspect_str    = r['aspect'][:20]
        sentiment_str = r['sentiment']
        conf_str      = f"{r['confidence']:.1f}%"
        clause_str    = r['clause'][:35] + ('...' if len(r['clause']) > 35 else '')
        print(f"  {aspect_str:<22} {icon} {sentiment_str:<8} {conf_str:<12} {clause_str}")

    # Overall summary
    sentiments = [r['sentiment'] for r in results]
    good_count = sentiments.count('Good')
    bad_count  = sentiments.count('Bad')

    print(f"\n  📊 Summary: {good_count} aspect(s) GOOD  |  {bad_count} aspect(s) BAD")

    if good_count > bad_count:
        print(f"  🎬 Overall Impression: POSITIVE review")
    elif bad_count > good_count:
        print(f"  🎬 Overall Impression: NEGATIVE review")
    else:
        print(f"  🎬 Overall Impression: MIXED review")

# ─────────────────────────────────────────────────────────────
# STEP 7 — DEMO with example reviews
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEMO — Aspect-Based Sentiment on Sample Reviews")
print("=" * 60)

demo_reviews = [
    # Classic contrast review
    "The story was brilliant and well written, but the action scenes were terrible and the CGI looked cheap.",

    # Multiple aspects
    "The acting was outstanding, the music was beautiful, but the plot was confusing and the pacing was too slow.",

    # All positive
    "Amazing direction, the story was gripping, and the cast delivered incredible performances throughout.",

    # All negative
    "Boring story, terrible acting, and the soundtrack was so annoying. The CGI effects were laughable.",

    # Subtle contrast
    "Despite the weak script, the cinematography was stunning and the emotional scenes were very moving.",
]

for review in demo_reviews:
    results = absa_predict(review)
    print_absa_results(review, results)
    print()

# ─────────────────────────────────────────────────────────────
# STEP 8 — LIVE ABSA PREDICTION
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LIVE ABSA — Type your own review!")
print("  Type 'quit' to exit")
print("=" * 60)

while True:
    print()
    user_input = input("Enter a movie review: ").strip()
    if user_input.lower() == 'quit':
        print("\n  Goodbye!")
        break
    if not user_input:
        print("  Please enter a review.")
        continue
    results = absa_predict(user_input)
    print_absa_results(user_input, results)