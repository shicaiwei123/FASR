import sys

sys.path.append('..')
import math
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern
from dsift import DsiftExtractor
import pickle
from lib.processing_utils import analysis, get_file_list
import datetime
import time
from lib.processing_utils import FaceDection, LandmarksDetection


def image2patch(image, patch_size, stride):
    """
    image:需要切分为图像块的图像
    patch_size:图像块的尺寸，如:(10,10)
    stride:切分图像块时移动过得步长，如:5
    """
    if len(image.shape) == 2:
        # 灰度图像
        imhigh, imwidth = image.shape
    if len(image.shape) == 3:
        # RGB图像
        imhigh, imwidth, imch = image.shape
    ## 构建图像块的索引
    range_y = np.arange(0, imhigh - patch_size[0], stride)
    range_x = np.arange(0, imwidth - patch_size[1], stride)
    if range_y[-1] != imhigh - patch_size[0]:
        range_y = np.append(range_y, imhigh - patch_size[0])
    if range_x[-1] != imwidth - patch_size[1]:
        range_x = np.append(range_x, imwidth - patch_size[1])
    sz = len(range_y) * len(range_x)  ## 图像块的数量
    if len(image.shape) == 2:
        ## 初始化灰度图像
        res = np.zeros((sz, patch_size[0], patch_size[1]))
    if len(image.shape) == 3:
        ## 初始化RGB图像
        res = np.zeros((sz, patch_size[0], patch_size[1], imch))
    index = 0
    for y in range_y:
        for x in range_x:
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            res[index] = patch
            index = index + 1

    return res


def mutil_moire_hist(img):
    # 转到255范围内
    img = np.uint8(img)

    # 提取多尺度特征,参数按照论文参数
    moire1 = local_binary_pattern(img, 8, 1)
    moire2 = local_binary_pattern(img, 24, 3)
    moire3 = local_binary_pattern(img, 40, 5)

    # 分别求直方图
    bins = np.arange(257)
    img_moire1_hist, bin1 = np.histogram(moire1, bins)
    img_moire2_hist, bin2 = np.histogram(moire2, bins)
    img_moire3_hist, bin3 = np.histogram(moire3, bins)
    moire_hist = img_moire1_hist + img_moire2_hist + img_moire3_hist
    return moire_hist


def feature_extract(img, ispatch=False):
    if ispatch:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 提取dsift特征
        begin = time.time()
        extractor = DsiftExtractor(8, 16, 1)
        (B, G, R) = cv2.split(img)
        image = cv2.merge((R, G, B))
        image = np.mean(np.double(image), axis=2)
        feaArr, positions = extractor.process_image(image)
        feaArr_flatten = feaArr.flatten()
        feature = list(feaArr_flatten)
        end = time.time()
        print("sift", end - begin)

        # 提取mmoire特征
        multi_moire_feaaature = mutil_moire_hist(img_gray)

        # 组合
        feature.extend(multi_moire_feaaature)
        # print("feature_len", len(feature))

        return feature
    else:
        img = cv2.resize(img, (224, 112))
        img_cut = image2patch(img, (32, 32), 16)
        for cut in img_cut:
            cut = np.uint8(cut)
            img_gray = cv2.cvtColor(cut, cv2.COLOR_RGB2GRAY)
            # 提取dsift特征
            extractor = DsiftExtractor(8, 16, 1)
            (B, G, R) = cv2.split(cut)
            image = cv2.merge((R, G, B))
            image = np.mean(np.double(image), axis=2)
            feaArr, positions = extractor.process_image(image)
            feaArr_flatten = feaArr.flatten()
            feature = list(feaArr_flatten)

            # 提取mmoire特征
            multi_moire_feaaature = mutil_moire_hist(img_gray)

            # 组合
            feature.extend(multi_moire_feaaature)
            # print("feature_len", len(feature))

        return feature


def data_make_single(img_dir, ispatch=False, balance=True, sample_interal=1):
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
    living_path_list = living_path_list[0:len(living_path_list):sample_interal]

    for file_path in living_path_list:
        img = cv2.imread(file_path)

        feature = feature_extract(img, ispatch=ispatch)
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

        feature = feature_extract(img, ispatch=ispatch)
        if feature is None:
            continue
        spoofing_feature.append(feature)
        spoofing_label.append(0)
        print(count)
        count += 1

    feature_all = living_feature + spoofing_feature
    label_all = living_label + spoofing_label

    return feature_all, label_all


def data_make(train_dir, test_dir, save_path, ispatch=False, balance=True, sample_interal=1):
    '''
    将照片数据,提取成feature 和label形式,并保存,为后续训练做准备
    :return:
    '''

    # 提取真人特征

    train_feature, train_label = data_make_single(train_dir,
                                                  ispatch=ispatch,
                                                  balance=balance,
                                                  sample_interal=sample_interal

                                                  )
    test_feature, test_label = data_make_single(test_dir,
                                                ispatch=ispatch,
                                                balance=balance,
                                                sample_interal=sample_interal
                                                )

    # 存储
    if not os.path.exists('../' + save_path):
        os.makedirs('../' + save_path)
    with open('../' + save_path + '/moire_train_feature', 'wb') as lf:
        pickle.dump(train_feature, lf)
    with open('../' + save_path + '/moire_train_label', 'wb') as ll:
        pickle.dump(train_label, ll)

    with open('../' + save_path + '/moire_test_feature', 'wb') as lf:
        pickle.dump(test_feature, lf)
    with open('../' + save_path + '/moire_test_label', 'wb') as ll:
        pickle.dump(test_label, ll)


def train_and_test(feature_path='.'):
    # 读取
    # 存储
    with open('../' + feature_path + '/moire_train_feature', 'rb') as lf:
        feature_train = pickle.load(lf)
    with open('../' + feature_path + '/moire_train_label', 'rb') as ll:
        label_train = pickle.load(ll)

    with open('../' + feature_path + '/moire_test_feature', 'rb') as lf:
        feature_test = pickle.load(lf)
    with open('../' + feature_path + '/moire_test_label', 'rb') as ll:
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

    with open('../' + feature_path + '/moire_model', 'wb') as lm:
        pickle.dump(model, lm)
    return model


def moire_model_test():
    '''
    用于模型的交叉测试,内部测试
    :return:
    '''
    face_detector = FaceDection(model_name='cv2')
    landmarks_detector = LandmarksDetection()
    with open('/home/shicaiwei/information_form_idcard/face_detection/reproduce/moire/intra_testing_bottom/moire_model',
              'rb') as lm:
        moire_model = pickle.load(lm)

    img_path = '/home/shicaiwei/data/liveness_data/intra_testing/test/living'
    isface = False
    label = 1
    file_list = get_file_list(img_path)
    count = 1
    true_num = 1
    time_begin = datetime.datetime.now()
    for file in file_list:
        img = cv2.imread(file)
        if img is None:
            continue
        if not isface:
            img = cv2.resize(img, (480, 640))
            face_img = face_detector.face_detect(img)
            if face_img is None:
                face_img = img

            landmarks = landmarks_detector.landmarks_detect(face_img)
            # 人脸裁剪
            key = landmarks
            left = key[0][0]  # 第一个特征点的纵坐标,也就是宽的方向
            right = key[16][0]  # 第17个特征点的纵坐标,也就是宽方形
            top = key[30][1]
            bottom = key[8][1]  # 第9个点的横坐标
            if left < 0:
                left = 0
            img_roi = face_img[top:bottom, left:right]
        else:
            face_img = img
            landmarks = landmarks_detector.landmarks_detect(face_img)
            # 人脸裁剪
            key = landmarks
            left = key[0][0]  # 第一个特征点的纵坐标,也就是宽的方向
            right = key[16][0]  # 第17个特征点的纵坐标,也就是宽方形
            top = key[30][1]
            bottom = key[8][1]  # 第9个点的横坐标
            if left < 0:
                left = 0
            img_roi = face_img[top:bottom, left:right]

        # img_roi = img
        # cv2.imshow("img_roi", img_roi)
        # cv2.waitKey(0)
        img_feature = feature_extract(img_roi, ispatch=False)
        if img_feature is None:
            continue
        result = moire_model.predict([img_feature])
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


def moire_feature_test():
    '''
    moire 特征测试
    :return:
    '''
    img = cv2.imread("/home/shicaiwei/data/liveness_data/live_data_select_jpg/2.png")
    img = cv2.resize(img, (480, 640))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_cut = image2patch(img, (32, 32), 16)
    for cut in img_cut:
        cut = np.uint8(cut)
        feature_cut = feature_extract(cut)
        print(len(feature_cut))


def moire_train(train_dir, test_dir, dataset_name='', sample_interal=1):
    save_path = dataset_name
    ispatch = False
    balance = True
    sample_interal = sample_interal
    data_make(train_dir=train_dir, test_dir=test_dir, save_path=save_path, ispatch=ispatch, balance=balance,
              sample_interal=sample_interal)
    train_and_test(feature_path=save_path)


if __name__ == '__main__':
    # moire_test()

    train_dir = "/home/shicaiwei/data/liveness_data/intra_testing_bottom/train"
    test_dir = "/home/shicaiwei/data/liveness_data/intra_testing_bottom/val"
    save_path = 'intra_testing_all'
    space = 'gray'
    moire_train(train_dir=train_dir, test_dir=test_dir, dataset_name=save_path, sample_interal=1)

    moire_model_test()
