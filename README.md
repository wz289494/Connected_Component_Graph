一、概述

这是一个将连接关系转化为邻接矩阵，再由邻接矩阵创造连通分量图的项目。

二、依赖

详见requirements.txt

终端安装:pip install -r requirements.txt

三、方法介绍

1、将相连关系保存到excel中，参考边列表.xlsx

2、将边关系转为邻接矩阵:

a = Twomode_Net('边列表.xlsx')

print(a.matrixdf)

a.show_data()

a.matrixdf.to_excel('邻接矩阵.xlsx')

3、将邻接矩阵中的联通分量进行拆取

= Con_component(a.matrixdf)

print(b.component_list)

（1）保存到图片文件夹

b.save_picfolder()

（2）展示数据

b.show_data()

（3）保存到excel文件夹

b.save_filefolder()

（4）绘制到最终图片

b.creat_finalpic()

![img.png](连通分量图.png)

四、更改

如果您想在此基础之上更改用途，只需要调整边列表即可