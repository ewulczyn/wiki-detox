from flask import Flask, render_template, jsonify, request, Response
import joblib
import argparse
import os
import json
from scoring_utils import get_comment
import time

app = Flask(__name__)
app.debug = True


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/wiki-talk/')
def main2():
    return render_template('index.html')


@app.route('/api')
def get_insult_scores():
    print(request.args)
    t1 = time.time()

    assert(request.args.get('model', '') in models)
    assert('text' in request.args or 'rev_id' in request.args)

    ret = {}

    if 'rev_id' in request.args:
        text = get_comment(request.args['rev_id'])
        ret['text'] = text
    else:
        text = request.args['text']
    
    if text:
        prob = models[request.args['model']].predict_proba([text])[0][1]
        ret['p'] = '%0.2f' % prob
    
    t2 = time.time()
    print('Total Time:', t2-t1)
    return jsonify(ret)


@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_paths', required=False,
    default='model_paths.json',
    help='path to json dictionary of model names and paths'
)

parser.add_argument(
    '--local', 
    action="store_true",
    help='run on local host '
)

args, unknown = parser.parse_known_args()
model_paths = json.load(open(args.model_paths))
models = {k : joblib.load(v) for k,v in model_paths.items()}


if __name__ == "__main__":
    if args.local:
        app.run(port=5002)
    else:
        app.run(host='0.0.0.0', port=8000)