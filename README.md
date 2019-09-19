粗暴的人脸识别
=

这是一个可能是最粗暴的人脸识别客流检测系统。找一个看上去有趣一些的项目。用于学习Python。
实际上最高深的内容都是C++完成的，和Python本身感觉上关系并不大。但是使用Python，真的是大大降低了编程的门槛。

该项目的第一版使用opencv实现人脸检测（就是网上很多的那个，25行完成人脸检测）。这个版本能够快速识别视频刘中的人脸信息，但是这些人脸opencv本身不会区分，很难做后续处理，姑且只能作为收集人脸识别的途径，当做为以后机器学习抓取样本。

第二版使用的是[Face Recognition](https://github.com/ageitgey/face_recognition)这个库，这个库基于dlib实现人脸识别功能，功能强大，还能使用Navidia显卡加速计算（我没有用）。通过Face Recognition库，实现人脸识别，去除**一分钟**内重复的人脸，然后**统计每个小时内检测到的人脸**，就算客流统计了，简单粗暴！

用到的库
```
python -m pip install --upgrade pip
pip install opencv-python
pip install face_recognition
pip install flask
```

人脸检测（已不使用）
==

检测到人脸，用绿色方框框起来，并保存为本地jpg。试了opencv的几个内置分类器感觉有两个比较好用：
1. haarcascade_frontalface_alt.xml 识别率比较高，稍慢
2. lbpcascade_frontalface.xml 识别率较低，会误识别，但巨快

真的很容易上手，简简单单的代码就能完成，看上去很炫的工作。得感谢前人的贡献呀。

```python
def detectFaces(video, outputPath, cascPath):
    num = 0   
    # 创建级联人脸识别器
    # haarcascade_frontalface_alt.xml
    # lbpcascade_frontalface.xml
    faceCascade = cv2.CascadeClassifier(cascPath)

    while True:
        # 抓取每一帧图像
        # ret， 返回值，表示视频是否读完
        # frame， 实际图像帧
        ret, frame = video.read()

        # 灰度处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 识别图像帧内人脸
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # 保存图像
            image_name = ('%s/%07d.jpg' % (outputPath, num))
            image = frame[y:y+h, x:x+w] 
            cv2.imwrite(image_name, image)
            num += 1

            # 根据识别出的人脸画框框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 展示图像
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()
```

人脸识别
==

真正的代码，使用Face Recognition识别人脸，对每张识别出的人脸生成特征码，根据特征码区分人脸。新检测到的人脸可以与已有的特征码对比，用来判断是否是新客户。将检测到的新人脸和特征码分别保存到各自的数组中，用来做后续的处理。

人脸识别宽容度(tolerance)设为**0.5**，该值越小精确度越高，但是性能消耗越大，越慢。实际使用下来感觉，人脸的无人率还是挺高的，特别是在人物运动，或者光照不充足而定情况下。只有在光照充足的正脸情况人的比较准确。

初始化的时候要读取一下已知人脸和特征码，因为没有做数据库，每次启动重新读一下吧

```python
print('1. 正在初始化人脸识别')
known_face_imgs = []
known_face_encodings = []
known_face_names = []

print('2. 正在读取已知人脸图片')
for file in os.listdir(outputPath):
        filePath = outputPath + '/' + file
        if (os.path.isfile(filePath) & file.endswith('.jpg')):
            known_face_names.append(os.path.splitext(file)[0])
            fr_image = face_recognition.load_image_file(filePath)
            known_face_imgs.append(fr_image)
            known_face_encodings.append(face_recognition.face_encodings(fr_image)[0])
```

把上面第一版的代码稍稍改一下，用Face Recognition代替opencv的分类器，主要就是两行

```python
# 识别图像中的人脸
face_locations = face_recognition.face_locations(rgb_small_frame)
# 提取人脸特征码
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
```

还要加一些其他逻辑：发现是新人脸才保存图片和特征码，如果是已知人脸就读出已知人脸的名称并显示

```python
import face_recognition
import cv2
import os
import sys

outputPath = 'output/image'
if len(sys.argv) >= 2:
    outputPath = sys.argv[1]

def detectFaces(video, outputPath):
    print('1. 正在初始化人脸识别')
    num = 0   
    known_face_imgs = []
    known_face_encodings = []
    known_face_names = []
    face_locations = []
    face_encodings = []
    face_names = []

    process_this_frame = True


    print('2. 正在读取已知人脸图片')
    for file in os.listdir(outputPath):
            filePath = outputPath + '/' + file
            if (os.path.isfile(filePath) & file.endswith('.jpg')):
                known_face_names.append(os.path.splitext(file)[0])
                fr_image = face_recognition.load_image_file(filePath)
                fr_encodings = face_recognition.face_encodings(fr_image)
                if (len(fr_encodings) > 0):
                    known_face_imgs.append(fr_image)
                    known_face_encodings.append(fr_encodings[0])

    print('3. 已知人脸解析完成')
    print('4. 开始解析摄像头视频流')

    while True:
        # 抓取每一帧图像
        # ret， 返回值，表示视频是否读完
        # frame， 实际图像帧
        ret, frame = video.read()

        # 等比缩小图片，提高处理速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # 处理图片： opencv 的图片是 bgr格式，转换成我们需要的 rgb格式
        rgb_small_frame = small_frame[:, :, ::-1]

        # 每2帧做一次人脸检测，提高效率
        if process_this_frame:
            # 每2帧做一次人脸检测，提高效率
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:

                # 对比人脸，找到最接近的人脸
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                # 模式1：判断是否与已知人脸匹配
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                else:
                    # 添加新人脸
                    name = ('%s%07d' % ('unknown', num))
                    known_face_names.append(name)
                    known_face_encodings.append(face_encoding)
                    num += 1

                # # 模式2：找到最接近的人脸
                # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                # best_match_index = np.argmin(face_distances)
                # if matches[best_match_index]:
                #     name = known_face_names[best_match_index]
                face_names.append(name)


        process_this_frame = not process_this_frame

        # 展示图像
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # 保存原始图像，原始尺寸
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 保存新人图片
            filePath = outputPath + '/' + name + '.jpg'
            if not os.path.exists(filePath):
                image = frame[top:bottom, left:right] 
                cv2.imwrite(filePath, image)

            # 根据识别出的人脸画框框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # 显示名称
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        
        # 输出播放
        cv2.imshow('粗暴的人脸识别', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    print('正在关闭摄像头')
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_captures = cv2.VideoCapture(0)
    detectFaces(video_captures, outputPath)


```



环境安装
===

_Face Recognition_ 库依赖 _dlib_ 库，因此windows上需要安装Visual Studio CMake环境，Linux需要gcc和g++才能够编译。安装完C++编译环境后直接运行后，直接使用pip就能安装该库

```
pip install face_recognition
```
Linux 安装 dlib (enable GPU)
```
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
```

Windows 安装 dlib (enable GPU)
1. 需要安装 VS 2015 的 C/C++ 编译器，亲测Window 7 安装2015，Windows 10安装2019，均可成功
2.如果之前安装了dlib,先卸载dlib, pip uninstall dlib
3.安装CUDA
4 安装CUDNN
CUDA下载: https://developer.nvidia.com/cuda-downloads
CUDNN下载: https://developer.nvidia.com/rdp/cudnn-download
注意:后者下载要注册英伟达开发帐号才可下载，下在完成后解压到CUDA Toolkit目录下就行
参考: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows

运行`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\demo_suite\bandwidthTest.exe` 及`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\demo_suite\deviceQuery.exe`测试，得到结果`Result=PASS`表示成功
5. 下载dlib并安装
```
python setup.py install
```
6. 测试结果
```
>python
Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 22:22:05) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import dlib
>>> print(dlib.DLIB_USE_CUDA)
True
>>> print(dlib.cuda.get_num_devices())
1
>>>
```


客流统计
==
客流统计十分之粗暴，包含以下三点规则
1. 统计每小时客流，本小时内，发现新人则计数+1
2. 日期变换，所有记录清零
3. 同一张人脸，在一分钟之内不重复计数

flow_counting
```python
import time

INIT_COUNTS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
                11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
daily_counts = INIT_COUNTS
ref_date = 20190101
ref_hour = 0
last_update_dict = {}

def initData():
    global daily_counts, ref_date, ref_hour, last_update_dict
    daily_counts = INIT_COUNTS
    ref_date, ref_hour = map(int, time.strftime("%Y%m%d %H").split())
    last_update_dict = {}

def garbageDataByDate():
    current_date, current_hour = map(int, time.strftime("%Y%m%d %H").split())
    if (current_date != ref_date):
        # 一旦日期切换，清空所有数据
        print('Date changed, clean all data')
        initData()
    elif (current_hour != ref_hour):
        # 每小时清空人脸记录，但不清计数
        print('Hour changed, clean last_update_dict')
        last_update_dict.clear()
    else:
        # do nothing
        pass
    return current_hour

def updateData(name, hour, time):
    last_update_dict[name] = time
    daily_counts[hour] += 1
    print("{} detected, current hour is {}, count is {}".format(name, hour, daily_counts[hour]))

# 如果一分钟之内没有重复检测，计数加一，并更新时间
def faceDetected(name):
    now = int(time.time())
    hour = garbageDataByDate()
    if (name in last_update_dict):
        lastUpdateTime = last_update_dict[name]
        # 一分钟之内用一张人脸不重复计数
        if ((now - lastUpdateTime) > 60):
            updateData(name, hour, now)
    else:
        # 发现新的客户，更新计数
        updateData(name, hour, now)

def getCounts():
    return daily_counts

initData()
```

输出视频
==
使用Flask将视频输出值html，并提供接口查询当前客流。该版本只允许一台浏览器接受视频流。

flask通过将一连串独立的jpeg图片输出来实现视频流，这种方法叫做motion JPEG，好处是延迟很低，但是成像质量一般。flask提供视频流是通过generator函数进行的，因此需要修改一下之前的输出视频的代码

```python
# # 原先使用opencv展示视频窗口
# cv2.imshow('粗暴的人脸识别', frame)

# 改为generator输出
ret, jpeg = cv2.imencode('.jpg', frame)
yield (b'--frame\r\n'
    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
```

视频服务主页以及需要的web service接口
```python
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
    # detectFaces是上面提到的人脸识别方法
    return Response(detectFaces(video_captures, outputPath, cascPath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  

@app.route('/total_count')  # 客流统计接口
def total_count():
     #统计文件夹下文件个数
    return Response(json.dumps(getCounts()),'text/plain') 

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)  
```


运行测试
==

```
python my-face-recognize/main.py D:\WorkSpace\github\opencv-face-recognize\output 
D:\WorkSpace\github\opencv-face-recognize\my-face-recognize
```

参考
== 
1. https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
2. https://blog.miguelgrinberg.com/post/video-streaming-with-flask

