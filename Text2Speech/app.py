#!/usr/bin/env python
# coding: utf-8


from flask import Flask, render_template, request, redirect
import pandas as pd
import pickle
from awsclient import polly_large_text

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('input.html', filename = None)
    

@app.route('/convert-text-to-speech',methods = ["POST"])
def result():
    request_data = dict(request.form)
    content = request_data.get("content")
    VoiceId = request_data.get("VoiceId")
    print("Content: ", content, "VoiceId: ", VoiceId)
    mp3_file_name = polly_large_text(content, VoiceId)
    print("Output: ", mp3_file_name)
    return render_template("input.html", filename = mp3_file_name, content = content)    
    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
    
    
    