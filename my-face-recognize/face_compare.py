import face_recognition
import cv2
import dlib
import time

def compare(source, target, model="hog"):
    """
    人脸1：1对比接口

    :param source 文件路径或文件对象,通常为身份证照片
    :param target 文件路径或文件对象，通常为摄像头截屏
    :return 返回对比距离，一般小于0.6可认为是同一人，严格模式可使用 <0.5, target中未识别人脸，返回99
    """

    if model != "hog" and  dlib.DLIB_USE_CUDA:
        model = "cnn"
        print("启用GPU加速，使用CNN模型")
    else:
        print("未启用GPU加速，使用HOG模型")

    start = time.perf_counter()

    # 读取身份证图片特征
    source_image = face_recognition.load_image_file(source)
    small_source = cv2.resize(source_image, (0, 0), fx=0.5, fy=0.5)
    source_locations = face_recognition.face_locations(small_source, model=model)
    source_encodings = [face_recognition.face_encodings(small_source, source_locations)[0]]

    # 读取摄像头图片特征
    # 等比缩小图片，提高处理速度
    target_image = face_recognition.load_image_file(target)
    small_target = cv2.resize(target_image, (0, 0), fx=0.25, fy=0.25)
    target_locations = face_recognition.face_locations(small_target, model=model)
    target_encodings = face_recognition.face_encodings(small_target, target_locations)

    # 对比
    if len(target_encodings) > 0:
        face_distances = face_recognition.face_distance(source_encodings, target_encodings[0])
        end  = time.perf_counter()
        print('图像处理时间：{:.2}秒'.format(end-start))

        for i, face_distance in enumerate(face_distances): # 只有一个
            return face_distance.max()
    return 99

if __name__ == '__main__':
    # 测试代码
    import numpy as np

    idcard_path = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\IMG_20190827_001.jpg'
    test_path1 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\zff.jpg' # 不戴眼镜 passed 
    test_path2 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\fengfan_zheng.jpg' # 戴眼镜 passed
    test_path3 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\qibo_sun.jpg' # 比较像 Reject
    test_path4 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\shan_lu.jpg' # 其他人 Rejected
    test_path5 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\jianfeng_dai.jpg'# 其他人 Rejected
    test_paths = [test_path1,test_path2,test_path3,test_path4,test_path5]
    with open(idcard_path, 'rb') as idcardImg:
        for test_path in test_paths:
            img = open(test_path, 'rb')
            face_distance = compare(idcardImg, img)
            print("对比结果:{:.4}".format(face_distance))

            # 显示图片
            idcard_cv_img = cv2.imread(idcard_path)
            test_cv_img = cv2.imread(test_path)
            idcard_cv_img = cv2.resize(idcard_cv_img, (400, 200))
            test_cv_img = cv2.resize(test_cv_img, (200, 200))
            hmerge = np.hstack((idcard_cv_img, test_cv_img)) #水平拼接
            font = cv2.FONT_HERSHEY_SIMPLEX

            if face_distance < 0.52:
                result_img = cv2.putText(hmerge, 'Approved', (150, 50), font, 2, (108,226,108), 2)
            else:
                result_img = cv2.putText(hmerge, 'Rejected', (150, 50), font, 2, (255,0,0), 2)
                
            cv2.imshow("Face Compare: {:.4}".format(face_distance), result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
