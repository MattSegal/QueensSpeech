from flask import Flask, render_template, request
from new_sentences import *

app = Flask(__name__)

@app.route('/words/')
def  words():
    default_num_words = 200
    try:
        words_arg = request.args.get('words')
        num_words = int(words_arg) if 3 < int(words_arg) < 400 else default_num_words
    except:
        num_words = default_num_words
    generated_words = generate_sentence(num_words)
    return render_template('queens_speech.html',words=generated_words)