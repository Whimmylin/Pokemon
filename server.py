from flask import Flask, jsonify, redirect, url_for, request, render_template
from flask_cors import CORS, cross_origin
from model import Model
from gevent.pywsgi import WSGIServer
import logging

app = Flask(__name__)
CORS(app)

vocab_file = 'data/dl-data/couplet/vocabs'
model_dir = 'data/dl-data/models/tf-lib/output_couplet'

m = Model(
        None, None, None, None, vocab_file,
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir=model_dir,
        restore_model=True, init_train=False, init_infer=True)

@app.route('/success/<nam>')
def success(nam):
   return '%s' % nam

@app.route('/chat/couplet/<in_str>')
def chat_couplet(in_str):
    if len(in_str) == 0 or len(in_str) > 50:
        output = u'您的输入太长了'
    else:
        output = m.infer(' '.join(in_str))
        output = ''.join(output.split(' '))
    print('上联：%s；下联：%s' % (in_str, output))
    return jsonify({'output': output})

@app.route('/test',methods = ['POST','GET'])
def test():
    if request.method == 'POST':
        user = request.form['mycouplet']
        print('%s' % user)
        output = m.infer(' '.join(user))
        output = ''.join(output.split(' '))
        result = {user: output}
        print('上联：%s；下联：%s' % (user, output))
        return render_template("one.html", user = user, output = output)
        # return redirect(url_for('success', nam=output))

http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()

if __name__ == '__main__':
   app.run(debug = True)