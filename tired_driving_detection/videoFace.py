import dlib
import cv2
import math
import numpy as np
import datetime
def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0]) #equalizeHist(in,out)
    cv2.merge(channels, ycrcb)
    img_eq=cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img_eq
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
# 计算两点距离
def calDistance(posA, posB):
    temp = pow(posA.x - posB.x, 2) + pow(posA.y - posB.y, 2)
    return math.sqrt(temp)

# 加载并初始化检测器
detector = dlib.get_frontal_face_detector()
# dlib的68点模型，使用训练好的特征预测器
predictor = dlib.shape_predictor("D:/study/shape_predictor_68_face_landmarks.dat")
# 视频路径
#camera = cv2.VideoCapture('D:/study/test.mp4')
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("cannot open camear")
    exit(0)
biggestRight = 0
biggestLeft = 0
smallestRight = 100
smallestLeft = 100
eyeCount = 0
mouthCount = 0
frameCount = 0
tiredList = []
count =0
count2=0
while True:
    # 读取帧
    ret, frame = camera.read()
    if not ret:
        break
    # 处理frame
    #frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #光照处理
    frame_ssr = hisEqulColor(frame)
    #无光照处理
    #frame_ssr=frame
    # if count2<300:
    #     frame_ssr=frame
    #     count2=count2+1
    #     cv2.imshow("Camera", frame_ssr)
    # else:
    frame_ssr = hisEqulColor(frame)
    # 检测脸部
    dets = detector(frame_ssr, 1)
    #print("Number of faces detected: {}".format(len(dets)))
    # 查找脸部位置
    for i, face in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} ".format(i, face.left(), face.top(), face.right(), face.bottom()))
        shape = predictor(frame_ssr, face)
        # 绘制脸部位置
        cv2.rectangle(frame_ssr, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)
        faceInMid = calDistance(shape.part(0), shape.part(27)) / calDistance(shape.part(27), shape.part(16))
        if faceInMid > 0.4 and faceInMid < 2.5:# 排除脸部旋转的情况
            count = count + 1
            tempLeft = (calDistance(shape.part(37), shape.part(41)) + calDistance(shape.part(38), shape.part(40))) / \
                       (calDistance(shape.part(36), shape.part(39)))
            tempRight = (calDistance(shape.part(43), shape.part(47)) + calDistance(shape.part(44), shape.part(46))) / \
                        (calDistance(shape.part(42), shape.part(45)))
            if count<30:
                cv2.putText(frame_ssr, "Preparing", (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if tempLeft > biggestLeft:
                    biggestLeft = tempLeft

                if tempRight > biggestRight:
                    biggestRight = tempRight

                if tempRight < smallestRight:
                    smallestRight = tempRight

                if tempLeft < smallestLeft:
                    smallestLeft = tempLeft
            else:
                if tempLeft-smallestLeft < (biggestLeft - smallestLeft) * 0.25 or tempRight - smallestRight < (biggestRight - smallestRight) * 0.25:
                    cv2.putText(frame_ssr, "closeEyes", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.imwrite('D:/photo/eye/eye' + str(eyeCount) + '.jpg', frame_ssr)  # 存储为图像
                    eyeCount = eyeCount + 1

                mouthOpen = calDistance(shape.part(62), shape.part(66)) / calDistance(shape.part(60), shape.part(64))
                if mouthOpen > 0.50:
                    cv2.putText(frame_ssr, "Yawning", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.imwrite('D:/photo/mouth/mouth' + str(mouthCount) + '.jpg', frame_ssr)  # 存储为图像
                    mouthCount = mouthCount + 1
        #显示68点
        #for i in range(68):
         # cv2.circle(frame_ssr, (shape.part(i).x, shape.part(i).y), 4, (0, 255, 0), -1, 8)
          #cv2.putText(frame_ssr, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.imshow("Camera", frame_ssr)

    if frameCount % 5 == 0 :
        cv2.imwrite('D:/photo/frame/frame' + str(frameCount) + '.jpg', frame_ssr)  # 存储为图像

    frameCount = frameCount + 1
    # 按esc逐帧判断
    #while True:
    #    key = cv2.waitKey(-1)
    #    if key == 27:
    #       break

    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()



