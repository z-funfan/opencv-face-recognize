import face_recognition
import cv2
import os
import dlib
import numpy as np
from flow_counting import faceDetected

def detectFaces(video, outputPath, cascPath, debug = False):
    print('1. 正在初始化人脸识别')
    num = 0   
    process_step = 3
    process_this_frame = 1
    known_face_imgs = []
    known_face_encodings = []
    known_face_names = []
    face_locations = []
    face_encodings = []
    face_names = []
    model = "hog"
    if dlib.DLIB_USE_CUDA:
        model = "cnn"
        print("启用GPU加速，使用CNN模型")
    else:
        print("未启用GPU加速，使用HOG模型")

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
        if (process_this_frame % process_step == 0):
            process_this_frame = 1
            face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:

                # 对比人脸，找到最接近的人脸
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                # # 模式1：判断是否与已知人脸匹配
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # 模式2：找到最接近的人脸
                if True in matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        face_names.append(name)
                else:
                    # 添加新人脸
                    name = ('%s%07d' % ('unknown', num))
                    known_face_names.append(name)
                    known_face_encodings.append(face_encoding)
                    num += 1


        process_this_frame += 1

        # 展示图像
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # 更新计数
            faceDetected(name)

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
        if (debug):
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yield frame
        else: 
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    # When everything is done, release the capture
    print('正在关闭摄像头')
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    outputPath = '../output/image'
    video_captures = cv2.VideoCapture(0)
    frameGenerator = detectFaces(video_captures, outputPath, '', True)
    for frame in frameGenerator:
        cv2.imshow('Face Recognize', frame)
