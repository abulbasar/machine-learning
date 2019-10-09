#!/usr/bin/env python
# coding: utf-8


import boto3
import uuid
import json
import shutil
import uuid
import os


polly = boto3.client("polly")
s3 = boto3.resource("s3")

class PollyService:
    
    polly = boto3.client("polly")
    s3 = boto3.resource("s3")
    
    # Voices: Aditi,Raveena,Ivy,Joanna,Kendra,Kimberly,Salli,Joey,Justin,Matthew
    # LanguageCode: 'en-IN'|'en-US'
    args = {
        "OutputFormat": "mp3",
        "VoiceId": "Aditi",
        "Text": None,
        "Engine": "standard",
        "LanguageCode": 'en-IN'

    }
    
    def __init__(self, output_location:str, save_to_s3:bool, VoiceId:str = "Aditi"):
        self.save_to_s3 = save_to_s3
        self.output_location = output_location
        self.VoiceId = VoiceId
        
        if output_location is None:
            raise Exception("Specify output location")

        if save_to_s3:
            print("Initializing bucket")
            self.bucket = s3.Bucket(output_location)
        else:
            self.local_path = output_location
    
    def _save_to_local(self, filename, data):
        with open(filename, "wb") as f:
            f.write(data)
            print("Written the output to a file - " + filename)
    
    def _save_to_s3(self, filename, data):
        bucket.put_object(Key = "audios/" + filename, Body = data)
    
    def synthensize(self, text):   
        message_id = str(uuid.uuid1())
        self.args["Text"] = text
        self.args["VoiceId"] = self.VoiceId

        response = polly.synthesize_speech(**self.args)
        data = response["AudioStream"].read()
        
        if self.save_to_s3:
            self.bucket.put_object(Key = "audios/{0}.mp3".format(message_id), Body = data)
            self.bucket.put_object(Key = "inputs/{0}.txt".format(message_id), Body = text.encode("utf-8"))
            print("Saved the output to " + self.output_location)
        else:
            filename = self.output_location + "/" + message_id + ".mp3"
            self._save_to_local(filename, data)
            print("Saved the output to " + self.output_location)
        return message_id

def split_into_chunks(content, limit = 3000):
    import spacy
    nlp = spacy.load('en_core_web_sm')
    text_sentences = nlp(content)
    sentences = list(text_sentences.sents)
    sentences = [sentence.text for sentence in sentences]
    """
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) > limit:
            chunks.append(chunk)
            chunk = sentence
        else:
            chunk += sentence
    if len(chunk)>0:
        chunks.append(chunk)
    """
    print("Input content size (chars): ", len(content))
    print("Number of sentences: ", len(sentences))
    #print("Number of chunks: ", len(chunks))
    return sentences

def concat_mp3(files, destination_file, del_source = False):
    destination = open(destination_file, 'wb')
    for filename in files:
        print("Reading file: ", filename)
        shutil.copyfileobj(open(filename, 'rb'), destination)
        if del_source:
            os.remove(filename)
    destination.close()


def polly_large_text(content, VoiceId):
    if not os.path.exists("static"):
        os.makedirs("static")

    polly_dir = "/tmp"
    polly_service = PollyService(polly_dir, False, VoiceId)
    chunks = split_into_chunks(content)
    mp3_file_ids = [polly_service.synthensize(chunk) for chunk in chunks]
    mp3s = [f"{polly_dir}/{file_id}.mp3" for file_id in mp3_file_ids]
    destination_file = str(uuid.uuid1()) + '.mp3'
    concat_mp3(mp3s, 'static/' + destination_file, True)
    with open("static/" + destination_file + ".txt", "w") as f:
        f.write(content)
    return destination_file


        
        
