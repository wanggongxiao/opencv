import cv2


def sort_contours(cnts, method="left-to-right"):
    revers = False

    if method == "right-to-left" or method == "bottom-to-top":
        revers = True
    if method == "left-to-right" or method == "bottom-to-top":
        i =1

    boundingBoxs = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxs) = zip(*sorted(zip(cnts,boundingBoxs),key=lambda  b:b[1][i],reverse=revers))

    return cnts,boundingBoxs


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized