import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import os

class cannyEdgeDetector:
    def __init__(self, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        self.img_final = None
    
    #高斯核 size*size
    def gaussian_kernel(self, size, sigma=1):
        size = int(size)//2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1/(2.0*np.pi*sigma**2)
        g = np.exp(-((x**2+y**2)/(2.0*sigma**2)))*normal#离散化的二维高斯函数
        return g

    def sobel_filters(self, img):
        # 索贝尔算子
        Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        # 偏导
        Ix = convolve(img, Kx)
        Iy = convolve(img, Ky)
        # 梯度幅值（因为梯度是个矢量，其大小称为幅值）和斜率
        G = np.hypot(Ix, Iy)# g=sqrt(Ix**2+Iy**2)
        G = G/G.max()*255 #修改范围
        G = G.astype(int)
        theta = np.arctan2(Iy, Ix)
        
        return (G, theta)

    #非极大值抑制：通俗意义上是指寻找像素点局部最大值，将非极大值点所对应的灰度值置为0
    #这样可以剔除掉一大部分非边缘的点
    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D*180./np.pi
        angle[angle<0]+=180
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    q=255
                    r=255
                    #angle 0
                    if (0<=angle[i, j]<22.5) or (157.5<=angle[i,j]<=180):
                        q=img[i, j+1]
                        r=img[i, j-1]
                    #angle 45
                    elif (22.5<=angle[i, j]<67.5):
                        q=img[i+1, j-1]
                        r=img[i-1, j+1]
                    #angle 90
                    elif (67.5<=angle[i,j]<112.5):
                        q=img[i+1, j]
                        r=img[i-1, j]
                    #angle 135
                    elif (112.5<=angle[i,j]<157.5):
                        q=img[i-1, j-1]
                        r=img[i+1,j+1]
                    
                    if img[i,j]>=q and img[i,j]>=r:
                        Z[i, j]=img[i,j]
                    else:
                        Z[i, j]=0
                except IndexError as e:
                    pass
        return Z

    #双重门槛
    def threshold(self, img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        highThreshold = int(img.max()*highThresholdRatio)
        lowThreshold = int(highThreshold*lowThresholdRatio)
        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)
        weak = np.int32(25)
        strong = np.int32(255)
        strong_i, strong_j = np.where(img>=highThreshold)
        zeros_i, zeros_j = np.where(img<lowThreshold)
        weak_i, weak_j = np.where((img<=highThreshold)&(img>=lowThreshold))
        
        res[strong_i, strong_j]=strong
        res[weak_i, weak_j]=weak
        
        return (res, weak, strong)

    # 滞后边缘跟踪：基于阈值结果，滞后包括将弱像素转换为强像素，当且仅当正在处理的像素周围的像素中的至少一个像素是强像素时
    def hysteresis(self, img, weak, strong=255):
        M, N=img.shape
        for i in range(1, M-1):
            for j in range(1, N-1):
                if img[i, j]==weak:
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                                or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                                or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                                img[i,j]=strong
                        else:
                            img[i,j]=0
                    except IndexError as e:
                        pass
        return img

    #rgb->gray， 这种转化方式实践效果好
    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1],rgb[:,:,2]
        gray = 0.2989*r+0.5870*g+0.1140*b
        return gray

    def detect(self, src):
        img = mpimg.imread(src)
        img = self.rgb2gray(img)
        img_smoothed = convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
        gradientMat, thetaMat = self.sobel_filters(img_smoothed)
        nonMaxImg = self.non_max_suppression(gradientMat, thetaMat)
        thresholdImg, weak, strong = self.threshold(nonMaxImg)
        return self.hysteresis(thresholdImg, weak, strong)


if __name__=="__main__":
    print("Enter the path:")
    c=input()
    if not os.path.exists(c):
        print("Path not exists.")
    else:
        print("waiting...")
        detector = cannyEdgeDetector(sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
        res = detector.detect(c)
        plt.imshow(res, "gray")
        plt.show()