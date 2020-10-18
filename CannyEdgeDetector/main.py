#批量转化文件
from canny_edge_detector import cannyEdgeDetector as ced
import os
import matplotlib.image as mpimg

src = "originImg" # 源文件夹
output = "sketch" # 输出文件夹
if not os.path.exists(src):
    print("Path not exists.")
else:
    print("----------START-------------")
    fileList = os.listdir(src)
    detector = ced(sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
    for i in fileList:
        print(i)
        res = detector.detect(src+"/"+i)
        mpimg.imsave("sketch/"+i, res, cmap="gray")
