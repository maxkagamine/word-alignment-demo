#!/usr/bin/env python3
'''
Small server to allow for running the aligner from the visualization page.
'''
from align import align as wsp_align
from awesome import align as awesome_align
from flask import Flask, send_file, request

app = Flask(__name__)

@app.get('/')
def index():
  return send_file('visualize.html')

@app.post('/align')
def align():
  method = request.json.get('method')
  from_language = request.json.get('fromLanguage')
  from_text = request.json.get('fromText')
  to_language = request.json.get('toLanguage')
  to_text = request.json.get('toText')
  awesome_model = request.json.get('awesomeModel')
  wsp_threshold = float(request.json.get('wspThreshold'))
  wsp_symmetric = bool(request.json.get('wspSymmetric'))
  wsp_symmetric_mode = request.json.get('wspSymmetricMode')

  match method:
    case 'awesome':
      result = awesome_align(from_language, from_text, to_language, to_text, awesome_model)
    case _:
      result = wsp_align(
        from_language,
        from_text,
        to_language,
        to_text,
        wsp_threshold,
        wsp_symmetric,
        wsp_symmetric_mode)

  return {
    'result': ','.join(str(i) for i in result)
  }

if __name__ == '__main__':
  app.run()
