#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request, url_for, redirect, flash, send_from_directory
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
import json
import io
import os
from pathlib import Path
import time
import codecs
from collections import defaultdict
import numpy as np
import spacy
from absl import logging
import tensorflow_hub as hub
import tensorflow as tf
import os
import pytextrank
import textacy
import re
from unsupervised_lookup import import_text, final_results
from objects import Document, return_embedder
import pickle
import docx2txt



# __________ APP PART START ____________

ALLOWED_EXTENSIONS = {'txt', 'docx'}
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'itStillRemainsASecret'
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, 'uploaded-files/')
app.config['TEXT_FOLDER'] = os.path.join(APP_ROOT, 'data/text/')
app.config['PICKLE_FOLDER'] = os.path.join(APP_ROOT, 'data/pickle-jar/')

print(app.config['UPLOAD_FOLDER'])

def available_files():
    basepath = Path(app.config['PICKLE_FOLDER'])
    files_in_pickle = [entry.name.rsplit('.', 1)[0] for entry in basepath.iterdir() if
                       entry.is_file() and entry.name.rsplit('.', 1)[1] == 'pkl']

    basepath = Path(app.config['TEXT_FOLDER'])
    files_in_text = [entry.name.rsplit('.', 1)[0] for entry in basepath.iterdir() if
                     entry.is_file() and entry.name.rsplit('.', 1)[1] == 'txt']
    return [f for f in files_in_pickle if f in files_in_text]

def pickle_new_doc(filename, sents, titles=[], defs=[]):
    first_time = time.time()
    d = Document(sents, titles, defs, embed)
    arr = [d[i] for i in range(len(d))]
    last_time = time.time()

    print(os.path.join(app.config['PICKLE_FOLDER'], filename))

    f = open(os.path.join(app.config['PICKLE_FOLDER'], filename), 'wb')
    pickle.dump(arr, f)
    f.close()
    print("Saved the .pkl file after")
    print("--- %.2f seconds ---" % (time.time() - last_time))
    print("Total time to pickle new file")
    print("--- %.2f seconds ---" % (time.time() - first_time))


class QueryForm(FlaskForm):
    query = StringField('Query')
    search = SubmitField('Search')


@app.route('/', methods=['GET', 'POST'])
def form():
    form = QueryForm()
    if request.method == 'POST':
        multiselect = request.form.getlist('mymultiselect')
        query = request.form['query'].strip()

        if query and multiselect:
            return return_result(query=query, files=multiselect)
        else:
            flash("Please type a query and select one or more files to search")
            return redirect(request.url)

    return render_template('get_query.html', form=form, available_files=available_files())


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('Please select a file to upload')
            return redirect(request.url)

        elif not allowed_file(file.filename):
            flash('Only .txt and .doc files are allowed')
            return redirect(request.url)

        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            text = docx2txt.process(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            text = [t.strip() + '\n' for t in text.split('\n') if t.strip()]
            with open(os.path.join(app.config['TEXT_FOLDER'], filename.rsplit('.', 1)[0] + '.txt'), 'w') as f:
                f.writelines(text)
            f.close()

            pickle_new_doc(filename.rsplit('.', 1)[0] + '.pkl', text, titles=[], defs=[])
            return redirect(url_for('form'))
    return render_template('upload.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/result/')
def return_result(query, files):
    titles = []
    sents = []
    arr = []
    clause2file = {}
    for file in files:
        sents += import_text(app.config['TEXT_FOLDER'] + file + '.txt')
        arr += pickle.load(open(app.config['PICKLE_FOLDER'] + file + '.pkl', 'rb'))
        for l in import_text(app.config['TEXT_FOLDER'] + file + '.txt'):
            clause2file[l] = file

    print(len(sents))
    print(len(arr))
    global all_lines
    all_lines = sents[:]
    last_time = time.time()
    results = final_results(query, arr, sents, titles, embed, n=25)

    which_file = [clause2file[r] for r in results]
    print('size of which_file is {}'.format(len(which_file)))
    print(which_file)

    print("Returned results after")
    print("--- %.2f seconds ---" % (time.time() - last_time))
    
    return render_template('return_result.html', query=query, results=results, which_file=which_file)


@app.route('/<path:query>+<path:result>/')
def result_with_view(query, result):
    return render_template('result_view.html', query=query, result=result, all_lines=all_lines)


# __________ APP PART END ____________



if __name__ == '__main__':
    
    print('\n------- NOW STARTING PROGRAM -------\n')
    embed = return_embedder()
    all_lines = []
    print('\n------- NOW STARTING SERVER -------\n')
    app.run(debug=True)
