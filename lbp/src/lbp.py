import sys
sys.path.append('..')
sys.path.append('./lbp')

import math
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
import pickle
import datetime
import os
print (os.path.abspath('.'))

from lib.processing_utils import get_file_list, FaceDection

face_detector = FaceDection(model_name='cv2')


def image_seg(img, n, display=False):
    '''
    将单通道图像等分成n份。n为1,4,9，。。
    :param img: 输入图像
    :return:
    '''

    # 判断n是不是完全平方数
    n_sqrt = math.sqrt(n)
    n_sqrt = int(n_sqrt)
    n_temp = n_sqrt * n_sqrt
    if n_temp != n:
        print("n is not exact square number")
        return 1

    # 图片分割
    img_shape = img.shape
    w = img_shape[0]
    h = img_shape[1]

    # 垂直分割，成n_sqrt个竖型
    w_remainder = w % 3
    h_remainder = h % 3
    img = img[0:w - w_remainder, 0:h - h_remainder]
    img_hsplit = np.hsplit(img, n_sqrt)

    # 对垂直分割后每一个部分水平分割
    img_seg = []
    for i in img_hsplit:
        img_vsplit = np.vsplit(i, n_sqrt)
        img_seg = img_seg + img_vsplit

    # 显示
    if display:
        for i in img_seg:
            cv2.imshow("seg", i)
            cv2.waitKey(0)
    return img_seg


def get_lbp_hist(img_signal):
    lbp_hist = []
    seg = img_signal
    seg = np.uint8(seg)
    seg_lbp = local_binary_pattern(seg, 8, 1, method='nri_uniform')

    # 求直方图

    bins = np.arange(60)
    seg_lbp_hist, bin = np.histogram(seg_lbp, bins)
    for i in seg_lbp_hist:
        lbp_hist = lbp_hist + [i]
    return lbp_hist


def feature_extract(img, space='gray', split_num=9, isface=False):
    '''
    输入一张图片，求指定空间的lbp特征
    :param img:
    :param space:
    :return:
    '''

    # 判断是否正确读到图像
    if img is None:
        print("img is None")
        os._exit(0)

    if isface:
        face_roi = img

        # cv2.imshow("face_roi",face_roi)
        # cv2.waitKey(0)
    else:
        # 人脸检测
        face_roi = face_detector.face_detect(img)
        if face_roi is None:
            face_roi = img
        # cv2.imshow("face_roi",face_roi)
        # cv2.waitKey(0)

    # 人脸分割

    # 颜色空间转换
    # 图像空间转换
    (B, G, R) = cv2.split(face_roi)  # 提取R、G、B分量
    y = 0.257 * R + 0.564 * G + 0.098 * B + 16
    cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
    cr = 0.439 * R - 0.368 * G - 0.071 * B + 128
    #
    # img_YCrCb = cv2.cvtColor(self.a_face_picture, cv2.COLOR_RGB2YCR_CB)
    # y = img_YCrCb[:, :, 0] + 16
    # cb = img_YCrCb[:, :, 1]
    # Cr = img_YCrCb[:, :, 2]

    img_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)

    if space == 'gray':
        face_roi = gray
    elif space == 'red':
        face_roi = R
    elif space == 'green':
        face_roi = G
    elif space == 'blue':
        face_roi = B
    elif space == 'H':
        face_roi = h
    elif space == 'S':
        face_roi = s
    elif space == 'V':
        face_roi = v
    elif space == 'Cr':
        face_roi = cr
    elif space == 'Cb':
        face_roi = cb
    elif space == 'Y':
        face_roi = y
    # cv2.imshow("img", face_roi)
    # cv2.waitKey(0)

    # 分割
    img_seg = image_seg(face_roi, split_num)

    # LBP特征
    img_lbp_hist = []
    for seg in img_seg:
        seg_hist = get_lbp_hist(seg)
        # print(len(seg_hist))
        img_lbp_hist = img_lbp_hist + seg_hist
    return img_lbp_hist


def data_make_single(img_dir, space='gray', isface=False, balance=True):
    '''
    对原始图像进行处理得到特征并且保存,方便后续的处理
    :param img_path:
    :param data_class:
    :param space:
    :param isface:
    :return:
    '''

    '''初始化'''
    living_dir = os.path.join(img_dir, 'living')
    spoofing_dir = os.path.join(img_dir, 'spoofing')
    living_feature = []
    living_label = []
    spoofing_feature = []
    spoofing_label = []
    count = 1

    living_path_list = get_file_list(living_dir)
    for file_path in living_path_list:
        img = cv2.imread(file_path)
        if not isface:
            img = cv2.resize(img, (480, 640))
        feature = feature_extract(img, space=space, isface=isface)
        if feature is None:
            continue
        living_feature.append(feature)
        living_label.append(1)
        print(count)
        count += 1

    spoofing_path_list = get_file_list(spoofing_dir)
    if balance:
        balance_factor = int(np.floor(len(spoofing_path_list) / len(living_path_list)))
        if balance_factor == 0:
            balance_factor = 1
        spoofing_path_list = spoofing_path_list[0:len(spoofing_path_list):balance_factor]
    for file_path in spoofing_path_list:
        img = cv2.imread(file_path)
        if not isface:
            img = cv2.resize(img, (480, 640))
        feature = feature_extract(img, space=space, isface=isface)
        if feature is None:
            continue
        spoofing_feature.append(feature)
        spoofing_label.append(0)
        print(count)
        count += 1

    feature_all = living_feature + spoofing_feature
    label_all = living_label + spoofing_label

    return feature_all, label_all


def data_make(train_dir, test_dir, save_path, space='gray', isface=False, balance=True):
    '''
    将照片数据,提取成feature 和label形式,并保存,为后续训练做准备
    :return:
    '''

    # 提取真人特征

    train_feature, train_label = data_make_single(train_dir,
                                                  space=space,
                                                  isface=isface, balance=balance)
    test_feature, test_label = data_make_single(test_dir,
                                                space=space,
                                                isface=isface, balance=balance)

    # 存储
    if not os.path.exists('../' + save_path):
        os.makedirs('../' + save_path)
    with open('../' + save_path + '/lbp_train_feature', 'wb') as lf:
        pickle.dump(train_feature, lf)
    with open('../' + save_path + '/lbp_train_label', 'wb') as ll:
        pickle.dump(train_label, ll)

    with open('../' + save_path + '/lbp_test_feature', 'wb') as lf:
        pickle.dump(test_feature, lf)
    with open('../' + save_path + '/lbp_test_label', 'wb') as ll:
        pickle.dump(test_label, ll)


def lbp_test():
    img = cv2.imread("/home/shicaiwei/information_form_idcard/face_detection/photo-8/1.jpg")

    mlbp1 = local_binary_pattern(img[:, :, 1], 24, 3)
    mlbp2 = local_binary_pattern(img[:, :, 1], 40, 5)

    # mlbp=np.uint8(mlbp)
    # k=(2**8-1)/(2**24-1)
    # test=mlbp*1
    mlbp1 = np.uint8(mlbp1)
    mlbp2 = np.uint8(mlbp2)

    cv2.imshow("mlpb1", mlbp1)
    cv2.imshow("mlpb2", mlbp2)
    cv2.waitKey(0)

    # 判断是否正确读到图像
    if img is None:
        print("img is None")
        os._exit(0)

    # 特征提取
    feature = feature_extract(img, space='gray', split_num=9)
    print(len(feature))


def train_and_test(feature_path='.'):
    # 读取
    # 存储
    with open('../' + feature_path + '/lbp_train_feature', 'rb') as lf:
        feature_train = pickle.load(lf)
    with open('../' + feature_path + '/lbp_train_label', 'rb') as ll:
        label_train = pickle.load(ll)

    with open('../' + feature_path + '/lbp_test_feature', 'rb') as lf:
        feature_test = pickle.load(lf)
    with open('../' + feature_path + '/lbp_test_label', 'rb') as ll:
        label_test = pickle.load(ll)

    # 扰乱
    feature_train = np.array(feature_train)
    label_train = np.array(label_train)
    feature_test = np.array(feature_test)
    label_test = np.array(label_test)

    shuffle_ix = np.random.permutation(np.arange(len(feature_train)))
    feature_train = feature_train[shuffle_ix]
    label_train = label_train[shuffle_ix]

    # # 训练
    from sklearn.svm import SVC
    trainX = np.array(feature_train)
    trainY = np.array(label_train)
    model = SVC(kernel='linear', C=0.001, gamma=0.5, cache_size=2000, class_weight='balanced')
    model.fit(trainX, trainY)

    # 测试
    # 结果
    true_wrong = 0
    true_right = 0
    false_wrong = 0
    false_right = 0

    test_len = len(label_test)
    for i in range(test_len):
        feature = feature_test[i]
        label = label_test[i]
        result = model.predict([feature])
        if result == label:
            if label == 1:
                true_right += 1
            else:
                false_right += 1
        else:
            if label == 1:
                true_wrong += 1
            else:
                false_wrong += 1
    print(true_right, true_wrong, false_right, false_wrong)
    analysis(true_right, true_wrong, false_right, false_wrong)

    # 保存返回模型

    with open('../' + feature_path + '/lbp_model', 'wb') as lm:
        pickle.dump(model, lm)
    return model


def lbp_model_test():
    '''
    模型的内部测试和交叉测试
    :return:
    '''
    with open('/home/shicaiwei/information_form_idcard/face_detection/reproduce/lbp/intra_testing/lbp_model',
              'rb') as lm:
        lbp_model = pickle.load(lm)

    time_begin = datetime.datetime.now()
    img_path = '/home/shicaiwei/data/liveness_data/intra_testing/test/living'
    label = 1
    isface = False
    space = 'gray'
    file_list = get_file_list(img_path)
    file_list = sorted(file_list)
    count = 1
    true_num = 1
    for file in file_list:
        img = cv2.imread(file)
        if img is None:
            continue
        if not isface:
            img = cv2.resize(img, (480, 640))
            pass
        img_feature = feature_extract(img, space=space, isface=isface)
        if img_feature is None:
            continue
        result = lbp_model.predict([img_feature])
        print(result)
        count += 1
        if result == label:
            true_num += 1
        else:
            print(file)

    print(count, true_num, true_num / count)

    time_end = datetime.datetime.now()
    time_all = time_end - time_begin
    print("time_all", time_all.total_seconds())


def lbp_train(train_dir, test_dir, dataset_name=''):
    save_path = dataset_name
    space = 'gray'
    isface = False
    data_make(train_dir=train_dir, test_dir=test_dir, save_path=save_path, space=space, isface=isface)
    train_and_test(feature_path=save_path)


if __name__ == '__main__':

    train_dir = "/home/shicaiwei/data/liveness_data/intra_testing_face/train"
    test_dir = "/home/shicaiwei/data/liveness_data/intra_testing_face/val"
    save_path = 'intra_testing'
    space = 'gray'

    # 输入的图像是否是人脸图像,是的话,则不会再进行人脸检测,不是则会.
    isface = True

    # 先对所有图像进行特征提取,然后再进行模型的训练,可以在后期修改模型或模型参数的时候不需要重新提取特征.
    data_make(train_dir=train_dir, test_dir=test_dir, save_path=save_path, space=space, isface=isface)

    # 训练模型
    train_and_test(feature_path=save_path)

    # 测试模型
    lbp_model_test()

