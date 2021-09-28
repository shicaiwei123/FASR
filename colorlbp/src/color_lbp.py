import math
import numpy as np
import cv2

import os
from skimage.feature import local_binary_pattern

import pickle
from reproduce.utils import analysis
import shutil
import datetime

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


def get_lbp_color_hist(img_signal):
    lbp_color_hist = []
    seg = img_signal
    seg = np.uint8(seg)
    seg_lbp_color = local_binary_pattern(seg, 8, 1, method='nri_uniform')

    # 求直方图

    bins = np.arange(60)
    seg_lbp_color_hist, bin = np.histogram(seg_lbp_color, bins)
    for i in seg_lbp_color_hist:
        lbp_color_hist = lbp_color_hist + [i]
    return lbp_color_hist


def feature_extract(img, split_num=1, isface=False):
    '''
    输入一张图片，求指定空间的lbp_color特征
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
        # if face_roi.shape[0] < 270:
    else:
        # 人脸检测
        face_roi = face_detector.face_detect(img)

    # 统一尺寸之后,效果反而会变差.
    # face_roi=cv2.resize(face_roi,(224,224))

    # 人脸分割

    # 颜色空间转换
    # 图像空间转换
    (B, G, R) = cv2.split(face_roi)  # 提取R、G、B分量
    y = 0.257 * R + 0.564 * G + 0.098 * B + 16
    cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
    cr = 0.439 * R - 0.368 * G - 0.071 * B + 128

    img_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)

    img_list = []
    img_list.append(y)
    img_list.append(cr)
    img_list.append(cb)

    # LBP特征
    img_lbp_color_hist = []
    for seg in img_list:
        seg_hist = get_lbp_color_hist(seg)
        # print(len(seg_hist))
        img_lbp_color_hist = img_lbp_color_hist + seg_hist
    return img_lbp_color_hist


def data_make_single(img_dir, isface=False, balance=True):
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
        feature = feature_extract(img, isface=isface)
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
        feature = feature_extract(img, isface=isface)
        if feature is None:
            continue
        spoofing_feature.append(feature)
        spoofing_label.append(0)
        print(count)
        count += 1

    feature_all = living_feature + spoofing_feature
    label_all = living_label + spoofing_label

    return feature_all, label_all


def data_make(train_dir, test_dir, save_path, isface=False, balance=True):
    '''
    将照片数据,提取成feature 和label形式,并保存,为后续训练做准备
    :return:
    '''

    # 提取真人特征

    train_feature, train_label = data_make_single(train_dir,
                                                  isface=isface,
                                                  balance=balance)
    test_feature, test_label = data_make_single(test_dir,
                                                isface=isface,
                                                balance=balance)

    # 存储
    if not os.path.exists('../'+save_path):
        os.makedirs('../'+save_path)
    with open('../'+save_path + '/lbp_color_train_feature', 'wb') as lf:
        pickle.dump(train_feature, lf)
    with open('../'+save_path + '/lbp_color_train_label', 'wb') as ll:
        pickle.dump(train_label, ll)

    with open('../'+save_path + '/lbp_color_test_feature', 'wb') as lf:
        pickle.dump(test_feature, lf)
    with open('../'+save_path + '/lbp_color_test_label', 'wb') as ll:
        pickle.dump(test_label, ll)


def lbp_color_test():
    img = cv2.imread("/home/shicaiwei/information_form_idcard/face_detection/photo-8/1.jpg")

    mlbp_color1 = local_binary_pattern(img[:, :, 1], 24, 3)
    mlbp_color2 = local_binary_pattern(img[:, :, 1], 40, 5)

    # mlbp_color=np.uint8(mlbp_color)
    # k=(2**8-1)/(2**24-1)
    # test=mlbp_color*1
    mlbp_color1 = np.uint8(mlbp_color1)
    mlbp_color2 = np.uint8(mlbp_color2)

    cv2.imshow("mlpb1", mlbp_color1)
    cv2.imshow("mlpb2", mlbp_color2)
    cv2.waitKey(0)

    # 判断是否正确读到图像
    if img is None:
        print("img is None")
        os._exit(0)

    # 特征提取
    feature = feature_extract(img, split_num=9)
    print(len(feature))


def train_and_test(feature_path='.'):
    # 读取
    # 存储
    with open('../'+feature_path + '/lbp_color_train_feature', 'rb') as lf:
        feature_train = pickle.load(lf)
    with open('../'+feature_path + '/lbp_color_train_label', 'rb') as ll:
        label_train = pickle.load(ll)

    with open('../'+feature_path + '/lbp_color_test_feature', 'rb') as lf:
        feature_test = pickle.load(lf)
    with open('../'+feature_path + '/lbp_color_test_label', 'rb') as ll:
        label_test = pickle.load(ll)

    # # 划分数据集
    #
    # # 分离真假
    # label_array = np.array(label)
    # feature_array = np.array(feature)
    # mask_true = label_array > 0
    # mask_false = label_array <= 0
    # label_true = label_array[mask_true]
    # label_false = label_array[mask_false]
    # feature_true = feature_array[mask_true]
    # feature_false = feature_array[mask_false]
    #
    # # 分离测试和训练
    # feature_train = np.vstack((feature_true[0:train_num], feature_false[0:train_num]))
    # label_train = np.hstack((label_true[0:train_num], label_false[0:train_num]))
    #
    # feature_test = np.vstack((feature_true[train_num:-1], feature_false[train_num:-1]))
    # label_test = np.hstack((label_true[train_num:-1], label_false[train_num:-1]))

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
    model = SVC(kernel='linear', C=1, gamma=0.5, cache_size=2000, class_weight='balanced')
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

    with open('../'+feature_path+'/lbp_color_model', 'wb') as lm:
        pickle.dump(model, lm)
    return model


def colorlbp_color_model_test():
    with open(
            '/home/shicaiwei/information_form_idcard/face_detection/reproduce/colorlbp/intra_testing/lbp_color_model',
            'rb') as lm:
        lbp_color_model = pickle.load(lm)

    time_begin = datetime.datetime.now()
    img_path = '/home/shicaiwei/data/liveness_data/intra_testing/test/living'
    label=1
    isface = False
    space = 'gray'
    file_list = get_file_list(img_path)
    count = 1
    true_num = 1
    for file in file_list:
        img = cv2.imread(file)
        if img is None:
            continue

        if not isface:
            img = cv2.resize(img, (480, 640))

        # 特征提取
        img_feature = feature_extract(img, isface=isface)
        if img_feature is None:
            continue

        # 判定
        result = lbp_color_model.predict([img_feature])
        print(result)
        count += 1
        if result == label:
            true_num += 1
        else:
            print(file)
            save_path = '/home/shicaiwei/data/liveness_data/false_cross_test'
            # shutil.copy(file, save_path)
        # if count > 50:
        #     break

    print(count, true_num, true_num / count)

    time_end = datetime.datetime.now()
    time_all = time_end - time_begin
    print("time_all", time_all.total_seconds())


def colorlbp_color_model_train(train_dir, test_dir, isface, dataset_name=''):
    save_path = dataset_name
    balance = True
    data_make(train_dir=train_dir, test_dir=test_dir, save_path=save_path, isface=isface, balance=balance)
    train_and_test(feature_path=save_path)


if __name__ == '__main__':
    train_dir = "/home/shicaiwei/data/liveness_data/intra_testing_face/train"
    test_dir = "/home/shicaiwei/data/liveness_data/intra_testing_face/val"
    save_path = 'intra_testing'
    space = 'gray'

    # 输入的图像是否是人脸图像,是的话,则不会再进行人脸检测,不是则会.
    isface = True
    colorlbp_color_model_train(train_dir=train_dir, test_dir=test_dir, dataset_name=save_path, isface=isface)
    colorlbp_color_model_test()
