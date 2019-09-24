from flask import Flask, request, Response
import json
from face_compare import compare
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
MAX_DISTANCE = '0.52'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

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
            max_distance = eval(request.form.get('maxDistance', MAX_DISTANCE))
            distance = compare(refImg,targetImg)
            if distance < max_distance:
                suggest = 'approved'
            else:
                suggest = 'rejected'
            return restJsonResponse( {'distance': distance, 'suggestion': suggest}) 

        else:
            return restJsonResponse('文件名不合法，请检查文件名或重命名文件', -101) 
    return

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)  
