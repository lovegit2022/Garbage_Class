import paddle
import paddle.nn as nn
import numpy as np
import json
import random
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageEnhance
from paddle.io import Dataset
from paddle.optimizer import Adam
from paddle.vision.transforms import Compose, Resize, ToTensor, Normalize#导入依赖库

data_path = 'Mydata'
#定义数据集
class MyDataset(Dataset):
    def __init__(self, mode='train', transform=None):
        super(MyDataset, self).__init__()
        self.data = []
        self.transform = transform
        with open(f'{data_path}{mode}.txt') as f:
            for line in f.readlines():
                info = line.strip().split(' ')
                if len(info) > 0:
                    self.data.append(
                        [data_path+'/'+info[0].strip(), info[1].strip()])

    def __getitem__(self, idx):
        image_file, label = self.data[idx]
        img = Image.open(image_file).convert('RGB')
        img = np.array(img)
        # (Tensor(shape=[3, 227, 227], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        if self.transform is not None:
            img = self.transform(img)
        label = np.array([label], dtype="int64")
        return img, label

    def __len__(self):
        return len(self.data)
 #定义数据增强方式     
transform = Compose([Resize(size=(227, 227)), paddle.vision.transforms.RandomVerticalFlip(),
                    paddle.vision.transforms.ColorJitter(hue = 0.5, brightness = 0.5),
                    Normalize(mean=[127.5],std=[127.5],data_format='HWC'), ToTensor(data_format="CHW")])
transform1 = Compose([Resize(size=(227, 227)), Normalize(mean=[127.5],
                                                        std=[127.5],
                                                        data_format='HWC'), ToTensor(data_format="CHW")])
#划分数据集
train_dataset = MyDataset(mode='/train', transform=transform)
test_dataset = MyDataset(mode='/test', transform=transform1)
val_dataset = MyDataset(mode='/val', transform=transform)
#定义网络模型
resnet50 = paddle.vision.models.resnet50(num_classes=158)
model = paddle.Model(resnet50)#对模型进行封装，便于后续训练以及评估
model.load('work/save')#用于加载训练好的模型参数，若第一次训练则不运行此命令
#配置优化器超参数
optim = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
             parameters=model.parameters(), weight_decay = 0.0001)
#定义损失函数
model.prepare(optim,
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())
#进行模型训练
paddle.get_device()
paddle.set_device('gpu:0')
model.fit(train_dataset,
          val_dataset,
          epochs=10,
          batch_size=64,
          verbose=1)
#保存模型
model.save('work/save')
#评估模型
model.evaluate(test_dataset, batch_size=64, verbose=1)
#利用测试集进行推理预测
results = model.predict(test_dataset)
labels = []
for result in results[0]:
    lab = np.argmax(result)
    labels.append(lab)
print(labels)#打印标签
#取单张测试集图片进行预测结果的可视化展示
test = MyDataset(mode='/test1', transform=transform)
result1 = model.predict(test)
lab1 = np.argmax(result1[0])
lab1#打印预测结果

import pylab as pl
import matplotlib.font_manager as fm

test_path = '/home/aistudio/Mydata/test1.txt'
myfont = fm.FontProperties(fname=r'/home/aistudio/simkai.ttf') # 设置字体   
jetson_path = '/home/aistudio/Mydata/garbage_classification.json'
with open(jetson_path, 'r') as f1:
    load_dict = json.load(f1)
with open(test_path, 'r') as f2:
    img_path = f2.readline().strip().split(' ')
test_img_path = '/home/aistudio/Mydata/' + f'{img_path[0]}'
print('输入测试图片路径为：')
print(test_img_path)
clas = load_dict[f'{lab1}']#从字典中查找预测标签对应的垃圾种类
img = cv2.imread(test_img_path)
plt.imshow(img)
plt.title(f'预测：{clas}', fontproperties = myfont, fontsize=20)
