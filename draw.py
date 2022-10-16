import csv
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd

from scipy.interpolate import make_interp_spline

# 导入数据
train_accc = open("F:/WORK/code/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet/result/1resnet-acc.csv")  # 打开csv文件
train_acc = csv.reader(train_accc,delimiter=',')  # 读取csv文件

val_accc =open("F:/WORK/code/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet/result/mul_notrip_acc.csv")

val_accc1 =open("F:/WORK/code/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet/result/nodataset_dealwith_final_acc.csv")
val_acc = csv.reader(val_accc,delimiter=',')  # 读取csv文件
val_acc1 = csv.reader(val_accc1,delimiter=',')  # 读取csv文件

s = open("F:/WORK/code/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet/result/2triplet-resnet-acc.csv")
ss=csv.reader(s,delimiter=',')

def smooth_xy(lx, ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 20)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]



x = []
y = []

for i in train_acc:  # 从第1行开始读取
    x.append(float(i[1])) # 将第1列数据从第1行读取到最后一行赋给列表x
    y.append(float(i[2]))
# 坐标轴设置

#xy = smooth_xy(x,y)

q=[]
w=[]

for i in val_acc:  # 从第1行开始读取
    q.append(float(i[1])) # 将第1列数据从第1行读取到最后一行赋给列表x
    w.append(float(i[2]))

#qw = smooth_xy(q, w)

u=[]
z=[]

for i in val_acc1:  # 从第1行开始读取
    u.append(float(i[1])) # 将第1列数据从第1行读取到最后一行赋给列表x
    z.append(float(i[2]))

#uz=smooth_xy(u,z)

a=[]
b=[]

for i in ss:  # 从第1行开始读取
    a.append(float(i[1])) # 将第1列数据从第1行读取到最后一行赋给列表x
    b.append(float(i[2]))

#ab=smooth_xy(a,b)

#plt.plot(xy[0],xy[1],label="1",marker='o')
#plt.plot(qw[0],qw[1],label="2",marker='2')
#plt.plot(uz[0],uz[1],label="3",marker='^')
#plt.plot(ab[0],ab[1],label="3",marker='3')

plt.plot(x,y,label="1")
plt.plot(u,z,label="3")



plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(fontsize=16)

#plt.rcParams['savefig.dpi']=1024#像素
#plt.rcParams['figure.dpi']=1024#分辨率
#plt.savefig('F:/WORK/code/deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet/result/11.png')

plt.show()