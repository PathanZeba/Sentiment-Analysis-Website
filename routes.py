from flask import Blueprint, render_template, request
from .model import analyze_sentiment
from flask import send_from_directory
import os

bp = Blueprint('main', __name__)

@bp.route('/favicon.ico')
def favicon():
    """Return the favicon.ico image, which is used to represent the application in the browser's address bar and bookmarks
    the favicon.ico image is served from the static directory of the application package. the same image is used for all routes
"""
    return send_from_directory(os.path.join(bp.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft-icon')

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    sentiment = analyze_sentiment(text)
    return render_template('result.html', text=text, sentiment=sentiment)

