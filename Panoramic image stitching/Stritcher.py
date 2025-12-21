from pydoc import describe
import stat
import numpy as np
import cv2


class Stitcher:
    # 拼接函数
    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        # 获取图片
        (imageB,imageA) = images
        # 检测A，B图片的SIFT关键点和描述符
        """
        todo

        """



    

        return None
    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    def detectAndDescribe(self, image):
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 创建SIFT生成器
        describe = cv2.SIFT_create()
        # 检测关键点和描述符
        (Kps, features) = describe.detectAndCompute(image, None)
        # 将关键点转换为Numpy数组
        kps = np.float32([kp.pt for kp in Kps])
        return (kps, features)
    


    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.BFMatcher()
        # 匹配描述符
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches=[]
        for m in rawMatches:
            # 应用Lowe's ratio test
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
            
        # 检查是否有足够的匹配点
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # 计算单应矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)
        return None 