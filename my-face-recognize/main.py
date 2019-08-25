from flask import Flask, render_template, Response
import cv2
import sys
import os
import json

# from faceDetection import detectFaces
from face_detect_v2 import detectFaces
from flow_counting import getCounts

outputPath = 'output/image'
if len(sys.argv) >= 2:
    outputPath = sys.argv[1]
    
resourcePath = ''
cascPath = 'haarcascade_frontalface_alt.xml'
if len(sys.argv) >= 3:
    resourcePath = sys.argv[2]
    cascPath = resourcePath + '/' + cascPath

print("resourcePath: " + resourcePath)
app = Flask(__name__, template_folder = resourcePath)

@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')

@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    # 从默认摄像头获取视频流
    print("Starting camera...")
    print("Output Path: " + outputPath)
    video_captures = cv2.VideoCapture(0)
    return Response(detectFaces(video_captures, outputPath, cascPath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  

@app.route('/total_count')  # 客流统计接口
def total_count():
     #统计文件夹下文件个数
    return Response(json.dumps(getCounts()),'text/plain') 

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)  
