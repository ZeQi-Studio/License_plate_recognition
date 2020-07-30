import cv2 as cv
import numpy as np
import os

# def replace(img_path):
#     for path in os.listdir(img_path):
#         path2 = img_path + path + "/"
#         for pathd in os.listdir(path2):
#             path3 = path2 + pathd
#             img2 = Image.open(path3)
#             img2 = img2.convert('RGBA')  # 图像格式转为RGBA
#             pixdata = img2.load()
#             for y in range(img2.size[1]):
#                 for x in range(img2.size[0]):
#                     if pixdata[x, y][0] > 220:  # 红色像素
#                         pixdata[x, y] = (255, 255, 255, 255)  # 替换为白色，参数分别为(R,G,B,透明度)
#             img2 = img2.convert('RGB')  # 图像格式转为RGB
#             print("替换文件",pathd)
#             img2.save(path3)
#红色像素替换为白色


def separate_color_red(img):
    #颜色提取
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv = np.array([200, 200, 43])  # 提取颜色的低值
    high_hsv = np.array([280, 255, 255])  # 提取颜色的高值
    mask = cv.inRange(img, lowerb=lower_hsv, upperb=high_hsv)
    print("颜色提取完成")
    return mask


def salt(img, n):
    #椒盐去燥
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
        print("去燥完成")
        return img


def show(name, img):
    #显示图片
    cv.namedWindow(str(name), cv.WINDOW_NORMAL)
    cv.resizeWindow(str(name), 800, 2000)  # 改变窗口大小
    cv.imshow(str(name), img)


def lines(img):
    # 直线检测
    img2 = cv.Canny(img, 2, 40)  #边缘检测
    line = 4
    minLineLength = 50
    maxLineGap = 150
    # HoughLinesP函数是概率直线检测，注意区分HoughLines函数
    lines = cv.HoughLinesP(img2, 1, np.pi / 180, 120, lines=line, minLineLength=minLineLength,maxLineGap=maxLineGap)
    lines1 = lines[:,0,:]
    # line 函数勾画直线
    # (x1,y1),(x2,y2)坐标位置
    # (0,255,0)设置BGR通道颜色
    # 2 是设置颜色粗浅度
    for x1, y1, x2, y2 in lines1:
        cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return img


def contour(img):
    #检测轮廓
    # ret, thresh = cv2.threshold(cv2.cvtColor(img, 127, 255, cv2.THRESH_BINARY))
    image, contours, hier = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:  #遍历轮廓
        rect = cv.minAreaRect(c)  #生成最小外接矩形
        box_ = cv.boxPoints(rect)
        h = abs(box_[3, 1] - box_[1, 1])
        w = abs(box_[3, 0] - box_[1, 0])
        print("宽，高",w,h)
        #只保留需要的轮廓
        if (h > 400 or w > 400):
            continue
        if (h < 40 or w < 40):
            continue
        count += 1
        box = cv.boxPoints(rect)  # 计算最小面积矩形的坐标
        box = np.int0(box)  # 将坐标规范化为整数
        angle = rect[2]  #获取矩形相对于水平面的角度
        print("angle",angle)
        if angle > 0:
            if abs(angle) > 45:
                angle = 90 - abs(angle)
        else:
            if abs(angle) > 45:
                angle = (90 - abs(angle))
        # 绘制矩形
        cv.drawContours(img, [box], 0, (255, 255, 255), 1)
    print("轮廓数量", count)
    print("坐标",box)
    return img, box, angle


def rotate(img, angle):
    #旋转图片
    (h, w) = img.shape[:2]  #获得图片高，宽
    center = (w // 2, h // 2)  #获得图片中心点
    img_ratete = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(img, img_ratete, (w, h))
    return rotated


def cut1(img, box):
    #从轮廓出裁剪图片
    x1, y1 = box[1]  #获取左上角坐标
    x2, y2 = box[3]  #获取右下角坐标
    img_cut = img[y1+10:y2-10, x1+10:x2-10]  #切片裁剪图像
    return img_cut


def cut2(img, out_path, filed):
    #裁剪方格中图像并保存
    #＠config i_:i为裁剪图像的y坐标区域, j_:j为裁剪图像的x坐标区域

    if not os.path.isdir(out_path):  #创建文件夹
        os.makedirs(out_path)
    h, w, _ = img.shape  #获取图像通道
    print(h, w)
    s,i_ = 0,0
    #循环保存图像
    for i in range(h//12,h,h//12):
        j_ = 0
        for j in range(w//8,w,w//8):
            imgd = img[i_ + 5:i - 5, j_ + 5:j - 5]
            # img_list.append(img[i_+10:i-10,j_+10:j-10])
            out_pathd =  out_path+filed[:-4]+"_"+str(s)+".jpg"  #图像保存路径
            cv.imwrite(out_pathd,imgd)
            print("保存文件",out_pathd)
            s += 1
            j_ = j
        i_ = i

if __name__ == "__main__":
    img_path = "E:/机器学习/较复杂环境下车牌号的识别/test_picture/test.jpg"      #读取图像文件夹

    # 解决imread不能读取中文路径
    img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), flags=cv.COLOR_BGR2HSV)
    print("begin")
    cv.namedWindow("imageshow")
    show("img", img)
    # img_contours1,_,_ = contour(img)
    # show("img_contours1", img_contours1)
    img_separate = separate_color_red(img)  # 提取蓝色框先
    show("img_separate", img_separate)
    img_contours2, box, angle = contour(img_separate)  # 轮廓检测，获取最外层矩形框的偏转角度
    show("img_contours2", img_contours2)
    img_cut = cut1(img_contours2,box)
    show("img_cut", img_cut)

    # mediu = cv.medianBlur(img_separate, 3)  # 中值滤波,过滤除最外层框线以外的线条
    # img_lines = lines(mediu)  # 直线检测，补充矩形框线
    # show("mediu", mediu)
    # show("img_lines", img_lines)
    # img_contours, box, angle = contour(img_lines)  # 轮廓检测，获取最外层矩形框的偏转角度
    # print("角度", angle, "坐标", box)
    # img_rotate = rotate(img_lines, angle)  # 旋转图像至水平方向
    # img_contours, box, _ = contour(img_rotate)  # 获取水平矩形框的坐标
    #
    # img_original_rotate = rotate(img, angle)  # 旋转原图至水平方向
    # img_original_cut = cut1(img_original_rotate, box)  # 通过图像坐标从外层矩形框处裁剪原图



    # show("mediu", mediu)
    # show("img_lines", img_lines)
    # show("img_contours", img_contours)
    # show("img__rotate",img_rotate)
    # show("img_original_cut",img_original_cut)
    cv.waitKey(0)
    cv.destroyAllWindows()


