## Canny边缘检测算法
文件夹自己建，图片自己找

文件说明：
* __pycache__/ python所需文件
* originImg/ 存储待操作的图片
* sketch/ 存储提取的轮廓
* merge/ 存储原图+轮廓
* canny_edge_detector.py 包含canny检测算法的类，也可以单独使用。
* main.py 批量操作
* merge_origin&sketch.py 合并两个文件夹中名字相同的图片
* 推导过程.ipynb
* 高斯核中sigma的取值原理.jpg

参考资料：
* https://blog.csdn.net/Nickter/article/details/18138081?locationNum=8&fps=1
* 微信公众号 相约机器人 在Python中逐步检测Canny边缘 - 计算机视觉

测试效果：
    gif可以提取第一帧,merge会失败
    webp无法提取
    如果背景比较暗，效果不佳
