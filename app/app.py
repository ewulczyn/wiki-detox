from flask import Flask, render_template, jsonify, request, Response
import joblib
import argparse
import os
import json
from scoring_utils import get_comment
import time

import sys
sys.path.append(os.path.abspath('../src/modeling'))
from serialization import load_pipeline


app = Flask(__name__)
app.config.from_pyfile('detox.ini')
app.debug = app.config['DEBUG']



@app.route('/')
def main():
    return render_template('index.html')

def process_rev_id(request):
    ret = {} 
    text = get_comment(request.args['input'])

    if not text:
        ret['error'] = 'ERROR: invalid revision id'
    else:
        ret['text'] = text
        ret['p'] = apply_model(text, request)
    return ret


def process_text(request):
    ret = {} 
    text = request.args['input']
    ret['p'] = apply_model(text, request)
    return ret
    

def apply_model(text, request):
    mtype = request.args['model']
    model = model_data[mtype]['model']
    classes = model_data[mtype]['classes']
    probs = model.predict_proba([text])[0]
    probs = ['%0.2f' % p for p in probs]
    return list(zip(classes, probs))
    

def validate_params(request):

    if request.args.get('model', '') not in model_data.keys():
        return 'ERROR: model type not provided'
    elif 'input_type' not in request.args:
        return 'ERROR: input_type not provided'
    elif request.args['input_type'] not in ['rev_id', 'text']:
        return 'ERROR: input_type must be rev_id or text'
    elif 'input' not in request.args:
        return 'ERROR: model input not provided'
    elif len(request.args['input']) == 0:
        return 'ERROR: empty input'
    else:
        return None



@app.route('/api')
def get_insult_scores():
    print(request.args)

    validation_error = validate_params(request)

    if validation_error:
        ret = {'error': validation_error}
    elif request.args['input_type']  == 'rev_id':
        ret = process_rev_id(request)
    else:
        ret = process_text(request)
    
    return jsonify(ret)
    


@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response


model_data = json.load(open("model_paths.json"))

for k, v in model_data.items():
    model = load_pipeline('./models', v['name'])
    v['model'] = model
    print(model.predict_proba(['fuck']))


if __name__ == "__main__":
    app.run(host='0.0.0.0')