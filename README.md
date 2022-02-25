### Garbage_Class
finish the garbage classification with DL
### 一、项目背景介绍

垃圾分类在人们日常生活中扮演着很重要的角色，有利于垃圾的回收再利用，提高材料的利用率，降低污染。但是由于人们在日常生活中对此类经验的缺乏以及生活习惯的影响等，导致垃圾不能很好的进行分类，各类垃圾总是混杂在一起。因此，需要设计出一套能够自动进行垃圾分拣的系统，能够对不同的垃圾完成自动识别。

### 二、数据介绍

本数据集集华为云垃圾分类数据集、各垃圾分类公开数据集及网络爬虫等于一身，经机器、人工多重高质量清洗筛选整合而成。  
本数据拥有训练集：43685张；验证集：5363张；测试集：5363张；总类别数：158类。另外，本数据集格式为ImageNet格式，符合多数主流api接口。

本数据集集图片属性为（720，540，3），适用于有关垃圾分类相关场景的使用，覆盖了日常生活中绝大多数的垃圾类型，包含可回收垃圾、有害垃圾、厨余垃圾、其他垃圾四大垃圾分类及其下分的总共 <em>158</em> 类的细分垃圾类型，平均每个类别有344张图片，单个类别最多有1662张图片，最少有6张图片。  
数据集链接为：[垃圾分类数据集](https://aistudio.baidu.com/aistudio/datasetdetail/77996)

下载好数据集压缩包并解压后即可查看数据集，现从可回收垃圾、有害垃圾、厨余垃圾、其他垃圾中各选一张图片进行展示。
只需运行以下代码即可查看图片实例
```
import cv2
import matplotlib.pyplot as plt#引入依赖库
img_read_path = ['Mydata/0/1127.jpg', 'Mydata/1/1352.jpg', 'Mydata/8/1127.jpg', 'Mydata/96/1001.jpg']#存放图片读取路径
title = ['recyclable garbage', 'other garbage', 'kitchen waste', 'hazardous garbage']#存放图片所对应类别，显示在图像标题处
for i in range(len(img_read_path)):
    img = cv2.imread(img_read_path[i])
    plt.figure()
    plt.imshow(img)
    plt.title(title[i])#显示图像与标签
```

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/ff39275eeba046f0bb29e2f19093d95fd040542cdeef442b9f51743216d18a36" width = "45%" height = "45%" />

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/5fc8069abe094143a9242d72529058ed60aef781ff62455da6422962430d3d08" width = "45%" height = "45%" />

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/b54c5f7ae2a541cd9fb27b6ef5f6d8371e2645f791f943488aa5e1bdac6e96cc" width = "45%" height = "45%" />

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/3bb7e8dc9a004e0bb35b9da8df9ef40660611ac55a1b49f39b6a767e1205bda7" width = "45%" height = "45%" />

上述四张图片即为代码运行结果，依次为可回收垃圾、其他垃圾、厨余垃圾以及有害垃圾。

### 三、模型介绍

本次模型采用迁移学习的方法，将一个已经预训练的模型ResNet应用在目标数据集上进行微调，能够达到较快的收敛速度。
ResNet提出了残差学习方法来减轻训练深层网络的困难。在已有设计思路(BN, 小卷积核，全卷积网络)的基础上，引入了残差模块。每个残差模块包含 <em>两条</em> 路径，其中一条路径是输入特征的直连通路，另一条路径对该特征做两到三次卷积操作得到该特征的残差，最后再将两条路径上的特征相加。

残差模块如图1所示，左边是基本模块连接方式，由两个输出通道数相同的3x3卷积组成。右边是瓶颈模块(Bottleneck)连接方式，之所以称为瓶颈，是因为上面的1x1卷积用来降维(图示例即256->64)，下面的1x1卷积用来升维(图示例即64->256)，这样中间3x3卷积的输入和输出通道数都较小(图示例即64->64)。

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/44c7f07ba27d437f999b5f832b6178c8b4216e2b7aac4f5e95b460874be3d8b6" width = "45%" height = "45%" />

常见的ResNet网络结构见下图：
![](https://ai-studio-static-online.cdn.bcebos.com/96cdcfe976f844ebb7dba3918267a1b0754a75d4d42146b9a18c518a70c0e318)


ResNet相关资料参考：[PaddlePaddle2.0之图像分类（ResNet）](https://aistudio.baidu.com/aistudio/projectdetail/1197790)

### 四、模型训练

模型训练可以使用 <em>paddle.Model()</em> 对ResNet模型进行封装，便于后续进行模型的训练以及评估

进行模型训练之前需要定义好优化器、损失函数等。优化器采用自适应优化器，包括学习率，权重衰减等超参数。学习率的设置需要谨慎，过大过小都会影响模型的收敛速度，权重衰减参数实际上与正则化相关联，可以防止模型的过拟合。
损失函数设置为交叉熵，是分类问题中经典的损失函数，能够描述两种分布的差异。
准备好数据集、优化器以及损失函数后，就可以开始模型训练了，可以使用fit方法进行训练，只需要传入epoch以及批大小等超参数即可让模型自动训练。以上代码实现如下：
```
resnet50 = paddle.vision.models.resnet50(num_classes=158)

model = paddle.Model(resnet50)#模型封装
model.load('work/save')#用于加载以前训练好的模型，如果第一次训练则注释该命令

optim = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
             parameters=model.parameters(), weight_decay = 0.0001)#配置优化器

model.prepare(optim,
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())#准备训练
              
paddle.get_device()

paddle.set_device('gpu:0')
model.fit(train_dataset,
          val_dataset,
          epochs=10,
          batch_size=64,
          verbose=1)#设置epoch、批大小等超参数进行训练
model.save('work/save')#保存模型
```

因为使用了预训练模型，所以在训练过程中会较快收敛，本人训练时训练大约30轮左右就达到了80%的准确率，验证集达到60%准确率，如果还想继续提高模型的准确性，可以考虑修改学习率，正则化参数并采用更多的数据增强（翻转、裁剪等）手段进行训练。

### 五、模型评估

由于之前已经使用了 <em>model = paddle.Model(resnet50)</em> 语句对模型进行了封装，所以对模型进行评估时只需调用内置函数即可
```
results = model.predict(test_dataset)#利用测试集评估模型准确率
```

在大约30轮的训练之后，训练出来的模型能够在测试集上达到大约60%的准确率，已经能识别大部分垃圾的种类

为了展现模型效果，在最后也进行了可视化，即抽取一张图片利用训练好的模型进行推理预测，将预测的标签进行展示。相关代码实现如下：
```
test = MyDataset(mode='/test1', transform=transform)
result1 = model.predict(test)
lab1 = np.argmax(result1[0])
lab1#打印预测标签

import pylab as pl
import matplotlib.font_manager as fm#导入库

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
```

本代码读取了一张金属罐的图片进行预测，最后将预测标签与图像一并显示，可视化结果如下：
![](https://ai-studio-static-online.cdn.bcebos.com/301cb30e96e346eb831001dad1135bce6945ae058ad041888284488ea3d92190)


可以看到，模型正确预测输入样本的种类。

### 六、总结与升华

本项目算是计算机视觉领域的一个经典图像分类问题，主要在于对现有工具的使用，例如paddleclas以及paddlex等。且本项目的一个优点在于使用迁移学习的方法，这样能够使得模型收敛的时间远远小于初始化每个卷积层全连接层权重的方法，在时间成本上大大减少。在做此项目时曾有意对比过两种方法：采用预训练模型只需要训练30~40个epoch即可达到可观的loss以及准确率，而从头开始初始化所有权重的方法需要大约100epoch才能有较好的效果。

鉴于当前模型的准确性，还有很大可以继续提升的空间。如果想改善模型的话，可以使用更多的数据增强方式，例如随机裁剪，随机翻转一定角度等，尽可能的贴近真实情况可能遇到的各种数据，扩充已有的数据集，提高的模型的辨别能力。
同时，为了防止模型出现过拟合的情况，还需要使用一些正则化手段，在奔项目中使用了L2正则化的方法，后续可以在网络中添加一些dropout层，修改L2正则化的系数等。

### 七、个人总结

本人刚刚从事计算机视觉领域的研究，学业水平还是该领域的一名小学生，本次项目的完成也是在查询了诸多资料的基础之上才写出来的，非常感谢各位前辈的资料。之后也将继续深入该领域的研究，培养这方面的兴趣。
