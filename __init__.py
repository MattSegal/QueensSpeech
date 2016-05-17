from flask import Flask, Blueprint, render_template, request
from new_sentences import *

queens_speech = Blueprint('queens_speech', __name__,
            template_folder='templates', 
            static_folder='static', static_url_path = '/queens_speech/static')


@queens_speech.route('/words/')
def  words():
    default_num_words = 200
    try:
        words_arg = request.args.get('words')
        num_words = int(words_arg) if 3 < int(words_arg) < 400 else default_num_words
    except:
        num_words = default_num_words
    generated_words = generate_sentence(num_words)
    return render_template('queens_speech.html',words=generated_words)


if __name__ == "__main__":
    app = Flask(__name__)
    app.register_blueprint(queens_speech)
    app.run(host= '0.0.0.0',debug=True)
