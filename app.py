from flask import Flask, render_template, request

from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import re

import nltk
nltk.download('punkt')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/text-summarization", methods=['POST'])
def summarize():

    if request.method == 'POST':
        inputtext = request.form['inputtext_']

        inputtext = str(re.sub(' +', ' ', inputtext))

        sentence_count = len(re.split(r'[.!?]+', inputtext))

        # Instantiate TextRankSummarizer()
        summarizer = TextRankSummarizer()

        parser = PlaintextParser.from_string(inputtext, Tokenizer('english'))

        # Ranking sentences on the basis of TextRank algorithm and choosing top 2 sentences for summary
        summary_sentences = summarizer(parser.document, round(0.2*sentence_count))

        # Printing the summary
        summary = ""

        for sentence in summary_sentences:
            summary = summary + str(sentence)

    return render_template("output.html", data = {'summary': summary})

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
