import numpy as np
import argparse
import cv2
from pexpect.screen import screen


# 图像绘制
def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height/float(h)
        dim = (int(w*r),height)

    else:
        r = width / float(w)
        dim = (width,int(h*r))

    resizeed = cv2.resize(image,dim,interpolation=inter)

    return resizeed
def order_point(pts):
    # 一共有四个点
    rect = np.zeros((4,2),dtpy = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmin(diff)]

    return  rect

def four_point_transform(image,pts):
    # 获取坐标点
    rect = order_point(pts)
    (tl, tr, br, bl) = rect
    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped
def main():
    ## 设置参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the image")
    args = vars(ap.parse_args())

    # 读取需要处理的图片
    image = cv2.imread(args["image"])

    ratio = image.shape[0]/500.0
    orign = image.copy()
    # 改变图像尺寸
    image = resize(orign,height = 500)

    # 预处理图像
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv_show("grap",gray)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    edged = cv2.Canny(gray,75,200)
    cv_show("edged",edged)

    # 发现边框
    cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts,key = cv2.contourArea,reverse=True)[:5]

    # 遍历轮廓
    for c in cnts:
        # 计算近似轮廓
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx) == 4:
            screenCnt = approx

    # 透视变换
    warped = four_point_transform(orign, screenCnt.reshape(4, 2) * ratio)

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
if __name__ == "__main__":
    main()