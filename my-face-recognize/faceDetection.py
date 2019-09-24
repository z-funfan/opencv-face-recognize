import cv2
import sys

outputPath = 'output/image'
if len(sys.argv) >= 2:
    outputPath = sys.argv[1]
    
resourcePath = ''
cascPath = 'haarcascade_frontalface_alt.xml'
if len(sys.argv) >= 3:
    resourcePath = sys.argv[2]
    cascPath = resourcePath + '/' + cascPath


def detectFaces(video, outputPath, cascPath, debug = False):
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
        if (debug):
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


    # When everything is done, release the capture
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_captures = cv2.VideoCapture(0)
    detectFaces(video_captures, outputPath, cascPath, True)
