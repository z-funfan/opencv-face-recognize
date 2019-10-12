import json

from flask import Flask, Response, request
from werkzeug.utils import secure_filename
import argparse

from face_compare import compare

# 启动参数
parser = argparse.ArgumentParser(description="Face compare web service interface")
parser.add_argument('--port', required=False, default=5000, type=int, help='Web service port, default is 5000')
parser.add_argument('--model', required=False, default='hog', help='Face detect model, hog and cnn is available, default value is hog')
parser.add_argument('--threshold', required=False,default=0.6,type=float,help='Default face compare distance threshold, the smaller the more accurate, deafult is 0.6')
parser.add_argument('--conference', required=False,default=0.85,type=float,help='Default conference percentage of the result, once the result larger than the conference, return approved, deafult is 85%')
args = parser.parse_args()

UPLOAD_FOLDER = './uploads'
MAX_DISTANCE = args.threshold
MIN_ACONFERENCE = args.conference
MODEL = args.model
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def restJsonResponse(data, code=0):
    return Response(json.dumps({'code': code, 'data': data}),\
         mimetype='application/json') 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/compare', methods = ['POST'])  # 返回distance
def face_compare():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'reference' not in request.files:
            return restJsonResponse( '缺少参考图像', -99) 
               
        if 'target' not in request.files:
            return restJsonResponse( '缺少目标图像', -99) 

        refImg = request.files['reference']
        targetImg = request.files['target']

        # if user does not select file, browser also
        # submit an empty part without filename
        if refImg.filename == '' or targetImg.filename == '':
            return restJsonResponse( '没有选择图像', -100) 

        if refImg and allowed_file(refImg.filename) and \
            targetImg and allowed_file(targetImg.filename):

            # compare faces and return result 
            max_distance = eval(request.form.get('maxDistance', str(MAX_DISTANCE)))
            min_conference = eval(request.form.get('conference', str(MIN_ACONFERENCE)))
            conference = compare(refImg, targetImg, model=MODEL, face_match_threshold=max_distance)
            if conference > min_conference:
                suggest = 'approved'
            else:
                suggest = 'rejected'
            return restJsonResponse( {'conference': conference, 'suggestion': suggest}) 

        else:
            return restJsonResponse('文件名不合法，请检查文件名或重命名文件', -101) 
    return

if __name__ == '__main__':
    PORT = args.port
    app.run(host='0.0.0.0', debug=False, port=PORT)  
