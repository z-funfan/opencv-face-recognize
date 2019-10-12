import time
import math

from cv2 import resize, imread, imshow, putText, waitKey, destroyAllWindows, FONT_HERSHEY_SIMPLEX
from dlib import DLIB_USE_CUDA
import face_recognition

# rrefer to Calculating Accuracy as a Percentage
# https://github.com/ageitgey/face_recognition/wiki/Calculating-Accuracy-as-a-Percentage
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def compare(source, target, model="hog", face_match_threshold=0.6):
    """
    人脸1：1对比接口

    :param source 文件路径或文件对象,通常为身份证照片
    :param target 文件路径或文件对象，通常为摄像头截屏
    :return 返回对比距离，一般小于0.6可认为是同一人，严格模式可使用 <0.5, target中未识别人脸，返回99
    """

    if model != "hog" and DLIB_USE_CUDA:
        model = "cnn"
        print("启用GPU加速，使用CNN模型")
    else:
        print("未启用GPU加速，使用HOG模型")

    start = time.perf_counter()

    # 读取身份证图片特征
    source_image = face_recognition.load_image_file(source)
    if source_image.size > (200 * 1024):
        source_image = resize(source_image, (0, 0), fx=0.5, fy=0.5)
    source_locations = face_recognition.face_locations(source_image, model=model)
    source_encodings = [face_recognition.face_encodings(source_image, source_locations)[0]]

    # 读取摄像头图片特征
    # 等比缩小图片，提高处理速度
    target_image = face_recognition.load_image_file(target)
    if target_image.size > (100 * 1024):
        target_image = resize(target_image, (0, 0), fx=0.25, fy=0.25)
    target_locations = face_recognition.face_locations(target_image, model=model)
    target_encodings = face_recognition.face_encodings(target_image, target_locations)

    # 对比
    if len(target_encodings) > 0:
        face_distances = face_recognition.face_distance(source_encodings, target_encodings[0])
        end  = time.perf_counter()
        print('图像处理时间：{:.2}秒'.format(end-start))

        for i, face_distance in enumerate(face_distances): # 只有一个
            return face_distance_to_conf(face_distance.max(), face_match_threshold)
    return 0.0

if __name__ == '__main__':
    # 测试代码
    import numpy as np

    idcard_path = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\zp.bmp'
    test_path0 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\IMG_20190827_001.jpg' # 不戴眼镜 passed 
    test_path1 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\zff.jpg' # 不戴眼镜 passed 
    test_path2 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\fengfan_zheng.jpg' # 戴眼镜 passed
    test_path3 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\qibo_sun.jpg' # 比较像 Reject
    test_path4 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\shan_lu.jpg' # 其他人 Rejected
    test_path5 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\jianfeng_dai.jpg'# 其他人 Rejected
    test_paths = [test_path0,test_path1,test_path2,test_path3,test_path4,test_path5]
    with open(idcard_path, 'rb') as idcardImg:
        for test_path in test_paths:
            img = open(test_path, 'rb')
            conference = compare(idcardImg, img)
            print("对比结果:{:.4}".format(conference))

            # 显示图片
            idcard_cv_img = imread(idcard_path)
            test_cv_img = imread(test_path)
            idcard_cv_img = resize(idcard_cv_img, (204, 252))
            test_cv_img = resize(test_cv_img, (252, 252))
            hmerge = np.hstack((idcard_cv_img, test_cv_img)) #水平拼接
            font = FONT_HERSHEY_SIMPLEX

            if conference > 0.85:
                result_img = putText(hmerge, 'Approved', (150, 50), font, 2, (108,226,108), 2)
            else:
                result_img = putText(hmerge, 'Rejected', (150, 50), font, 2, (255,0,0), 2)
                
            imshow("Face Compare: {:.4}%".format(conference * 100), result_img)
        waitKey(0)
        destroyAllWindows()
