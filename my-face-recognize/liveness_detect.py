
from models.FeatherNet import FeatherNetB
from models.loss import FocalLoss
from collections import OrderedDict
from torchvision.transforms import transforms
from PIL import Image

import torch
import random
import numpy as np
import os

## Set random seeds ##
torch.manual_seed(14)
np.random.seed(14)
random.seed(14)

img_size = 224
ratio = 224.0 / float(img_size)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.14300402, 0.1434545, 0.14277956], std=[0.10050353, 0.100842826, 0.10034215])##accorcoding to casia-surf val to commpute
])

model = FeatherNetB()
# optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=0.0001)


path = os.path.join(os.getcwd(), 'model\_47_best.pth.tar')
device = torch.device('cpu')
checkpoint = torch.load(path, map_location=device)

state_dict =checkpoint['state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove 'module.' of dataparallel
    new_state_dict[name]=v


model.load_state_dict(new_state_dict)
# optimizer.load_state_dict(new_state_dict)
print("=> loaded checkpoint '.\model\_47_best.pth.tar' (epoch {})".format(checkpoint['epoch']))
model.eval()

def validate(image):
    global model, transform, device
    try:
        img = Image.open(image)
    except:
        img = image
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img = transform(img)

    img = np.array(img)  # 形状为(3, 224, 224)
    img = np.expand_dims(img, 0)  # 此时形状为(1, 3, 224, 224)
    with torch.no_grad():
        img_tensor = torch.tensor(img, dtype=torch.float32, device=device)
        output = model(img_tensor)
        soft_output = torch.softmax(output,dim=-1)
        preds = soft_output.to('cpu').detach().numpy()
        _,predicted = torch.max(soft_output.data, 1)
        predicted = predicted.to('cpu').detach().numpy()
        return predicted.max()

if __name__ == '__main__':
    from cv2 import  resize, imread, imshow, putText, waitKey, destroyAllWindows, FONT_HERSHEY_SIMPLEX

    test_path0 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\IMG_20190827_001.jpg'
    test_path1 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\zff.jpg' # 不戴眼镜 passed 
    test_path2 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\fengfan_zheng.jpg' # 戴眼镜 passed
    test_path3 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\zff-remake-video.jpg' # 比较像 Reject
    test_path4 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\shan_lu.jpg' # 其他人 Rejected
    test_path5 = 'D:\\sandbox-aw\\github\\opencv-face-recognize\\example\\jianfeng_dai.jpg'# 其他人 Rejected
    test_paths = [test_path0, test_path1,test_path2,test_path3,test_path4,test_path5]
    i = 0
    for test_path in test_paths:
        predicted = validate(test_path)
        print("对比结果:{}".format(predicted))

        # 显示图片
        idcard_cv_img = imread(test_path)
        font = FONT_HERSHEY_SIMPLEX

        if predicted < 0.52:
            result_img = putText(idcard_cv_img, 'Fake: ' + str(predicted), (0, 50), font, 2, (108,226,108), 2)
        else:
            result_img = putText(idcard_cv_img, 'Real ' + str(predicted), (0, 50), font, 2, (255,0,0), 2)
            
        imshow("Face Compare: {}".format(str(i)), result_img)
        i += 1
    waitKey(0)
    destroyAllWindows()
