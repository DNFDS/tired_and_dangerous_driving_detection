import numpy as np
import cv2
import math
import random
import sys

def singleScaleRetinex(img, sigma):

    _temp = cv2.GaussianBlur(img, (0, 0), sigma)
    guassian = np.where(_temp == 0, 0.01, _temp)
    img_ssr = np.log10(img + 0.01) - np.log10(guassian)

    for i in range(img_ssr.shape[2]):
        img_ssr[:, :, i] = (img_ssr[:, :, i] - np.min(img_ssr[:, :, i])) / (np.max(img_ssr[:, :, i]) - np.min(img_ssr[:, :, i])) * 255

    img_ssr = np.uint8(np.minimum(np.maximum(img_ssr, 0), 255))
    return img_ssr


def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    avgB = np.average(nimg[0])
    avgG = np.average(nimg[1])
    avgR = np.average(nimg[2])

    avg = (avgB + avgG + avgR) / 3

    nimg[0] = np.minimum(nimg[0] * (avg / avgB), 255)
    nimg[1] = np.minimum(nimg[1] * (avg / avgG), 255)
    nimg[2] = np.minimum(nimg[2] * (avg / avgR), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def Illumination_compensation(img):
    col=0;
    index=0;
    rows=img.shape[0]
    cols=img.shape[1]

def whiteBalance(img):

    rows = img.shape[0]
    cols = img.shape[1]

    final = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    avg_a = np.average(final[:, :, 1])
    avg_b = np.average(final[:, :, 2])

    for x in range(final.shape[0]):
        for y in range(final.shape[1]):
            l, a, b = final[x, y, :]
            l *= 100/255.0
            final[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            final[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)

    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)

    return final

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0]) #equalizeHist(in,out)
    cv2.merge(channels, ycrcb)
    img_eq=cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img_eq


def stretchImage(data, s=0.005, bins=2000):  # 线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]
    ht = np.histogram(data, bins);
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0;
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)


g_para = {}


def getPara(radius=5):  # 根据半径计算权重参数矩阵
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m


def zmIce(I, ratio=4, radius=300):  # 常规的ACE实现
    para = getPara(radius)
    height, width = I.shape
    zh, zw = [0] * radius + range(height) + [height - 1] * radius, [0] * radius + range(width) + [width - 1] * radius
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res


def zmIceFast(I, ratio, radius):  # 单通道ACE快速增强实现
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, ((width + 1) // 2, (height + 1) // 2))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)


def zmIceColor(I, ratio=4, radius=3):  # rgb三通道分别增强，ratio是对比度增强因子，radius是卷积模板半径
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res
import cv2
import numpy as np

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst


# 饱和函数
def calc_saturation(diff, slope, limit):
    ret = diff * slope
    if ret > limit:
        ret = limit
    elif (ret < (-limit)):
        ret = -limit
    return ret


def automatic_color_equalization(nimg, slope=10, limit=1000, samples=500):
    nimg = nimg.transpose(2, 0, 1)

    # Convert input to an ndarray with column-major memory order(仅仅是地址连续，内容和结构不变)
    nimg = np.ascontiguousarray(nimg, dtype=np.uint8)

    width = nimg.shape[2]
    height = nimg.shape[1]

    cary = []

    # 随机产生索引
    for i in range(0, samples):
        _x = random.randint(0, width) % width
        _y = random.randint(0, height) % height

        dict = {"x": _x, "y": _y}
        cary.append(dict)


    mat = np.zeros((3, height, width), float)

    r_max = sys.float_info.min
    r_min = sys.float_info.max

    g_max = sys.float_info.min
    g_min = sys.float_info.max

    b_max = sys.float_info.min
    b_min = sys.float_info.max

    for i in range(height):
        for j in range(width):
            r = nimg[0, i, j]
            g = nimg[1, i, j]
            b = nimg[2, i, j]

            r_rscore_sum = 0.0
            g_rscore_sum = 0.0
            b_rscore_sum = 0.0
            denominator = 0.0

        for _dict in cary:
            _x = _dict["x"]  # width
            _y = _dict["y"]  # height

            # 计算欧氏距离
            dist = np.sqrt(np.square(_x - j) + np.square(_y - i))
            if (dist < height / 5):
                continue;

            _sr = nimg[0, _y, _x]
            _sg = nimg[1, _y, _x]
            _sb = nimg[2, _y, _x]

            r_rscore_sum += calc_saturation(int(r) - int(_sr), slope, limit) / dist
            g_rscore_sum += calc_saturation(int(g) - int(_sg), slope, limit) / dist
            b_rscore_sum += calc_saturation(int(b) - int(_sb), slope, limit) / dist

            denominator += limit / dist

        r_rscore_sum = r_rscore_sum / denominator
        g_rscore_sum = g_rscore_sum / denominator
        b_rscore_sum = b_rscore_sum / denominator

        mat[0, i, j] = r_rscore_sum
        mat[1, i, j] = g_rscore_sum
        mat[2, i, j] = b_rscore_sum

        if r_max < r_rscore_sum:
            r_max = r_rscore_sum
        if r_min > r_rscore_sum:
            r_min = r_rscore_sum

        if g_max < g_rscore_sum:
            g_max = g_rscore_sum
        if g_min > g_rscore_sum:
            g_min = g_rscore_sum

        if b_max < b_rscore_sum:
            b_max = b_rscore_sum
        if b_min > b_rscore_sum:
            b_min = b_rscore_sum

    for i in range(height):
        for j in range(width):
            nimg[0, i, j] = (mat[0, i, j] - r_min) * 255 / (r_max - r_min)
            nimg[1, i, j] = (mat[1, i, j] - g_min) * 255 / (g_max - g_min)
            nimg[2, i, j] = (mat[2, i, j] - b_min) * 255 / (b_max - b_min)

    return nimg.transpose(1, 2, 0).astype(np.uint8)

frame = cv2.imread("D:/study/study.png")

frame_new = grey_world(frame)
frame_white = hisEqulColor(frame)
frame_ssr = automatic_color_equalization(frame)
dst = unevenLightCompensate(frame, 16)
result = np.concatenate([frame, dst], axis=1)

cv2.imshow("Camera", frame)
cv2.imshow("Grey_World", frame_new)
cv2.imshow("White_Balance", result)
cv2.imshow("SSR", frame_ssr)

while True:
    # 读取帧

    key = cv2.waitKey(1)
    if key == 27:
        break


