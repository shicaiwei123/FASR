import math
import numpy as np
import cv2
import os
# import pandas as pd
from scipy import stats
import pickle
from reproduce.utils import analysis
import datetime

from lib.processing_utils import get_file_list, FaceDection

face_detector = FaceDection(model_name='cv2')


def remove_spec(img, display=False):
    img = img[30:200, 30:200]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    (w, h) = img_gray.shape
    # mask = np.zeros((w,h)
    mask = img_gray.copy()
    mask[mask < 170] = 0
    mask[mask >= 170] = 255
    diff = img_gray.copy()
    spec = img_gray.copy()
    # spec[:,:]=0
    spec[mask > 0] = 255
    diff[mask > 0] = 0

    test = cv2.illuminationChange(img, mask, alpha=2, beta=2)
    test_gray = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    high_light = img_gray - test_gray
    high_light = np.uint8(high_light)

    if display:
        cv2.imshow("origin", img)
        cv2.imshow("test", test)
        cv2.imshow("high", high_light)
        cv2.imshow("spec", spec)
        cv2.imshow("diff", diff)
        cv2.waitKey(0)


def blur_detect2(img, display=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_soble_y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)
    abs_y = cv2.convertScaleAbs(img_soble_y)
    abs_y[abs_y < 30] = 0
    (w, h) = abs_y.shape
    edge_num = 0
    width = 0
    width_all = 0
    is_edge = False
    for i in range(w):
        for j in range(h):
            num = abs_y[i, j]
            if num > 0:
                is_edge = True
                width += 1
            else:
                if is_edge:
                    edge_num += 1
                    is_edge = False
                    width_all += width
                    width = 0
    blur = width_all / edge_num
    print("blur", blur)

    if display:
        cv2.imshow("sobel", abs_y)
        cv2.waitKey(0)

    return [blur]


def img_split(img, display=False):
    '''
    将图像分成镜面反射部分和漫反射两个部分
    :param img:
    :return:
    '''
    # 将图像格式转成RGB顺序
    (B, G, R) = cv2.split(img)
    img_rgb = cv2.merge((R, G, B))

    # 转写代码
    img_rgb = np.float64(img_rgb)
    img_rgb = img_rgb / 255
    total = np.sum(img_rgb, 2)
    (w, h, d) = img_rgb.shape
    sigma = np.zeros((w, h, d))
    for i in range(d):
        sigma[:, :, i] = img_rgb[:, :, i] / total

    where_are_nan = np.isnan(sigma)
    sigma[where_are_nan] = 0
    sigma_min = np.min(sigma, 2)
    sigma_max = np.max(sigma, 2)
    lam = np.ones((w, h, d)) / 3
    for i in range(d):
        test = (3 * (lam[:, :, i] - sigma_min))
        lam[:, :, i] = (sigma[:, :, i] - sigma_min) / (3 * (lam[:, :, i] - sigma_min))
    where_are_nan = np.isnan(lam)
    lam[where_are_nan] = 1 / 3

    if w > h:
        SIGMAS = 0.25 * h
    else:
        SIGMAS = 0.25 * w

    SIGMAR = 0.04
    THR = 0.03

    while True:
        sigma_max = np.float32(sigma_max)
        sigma_max_F = cv2.bilateralFilter(sigma_max, 5, 50, 50)
        s = np.sum(sigma_max_F - sigma_max > THR)
        if np.sum(sigma_max_F - sigma_max > THR) == 0:
            break

        # 取最大值迭代
        temp_sub = sigma_max - sigma_max_F
        sigma_max[temp_sub < 0] = sigma_max_F[temp_sub < 0]

    # 求镜面反射
    img_max = np.max(img_rgb, 2)
    den = 1 - 3 * sigma_max

    spec = (img_max - sigma_max * total) / den
    spec[den == 0] = den.max()
    spec[spec < 0] = 0
    spec[spec > 1] = 1

    # 求漫反射
    diffuse = np.zeros((w, h, d))
    for i in range(d):
        diffuse[:, :, i] = img_rgb[:, :, i] - spec
    diffuse[diffuse < 0] = 0
    diffuse[diffuse > 1] = 1
    (R, G, B) = cv2.split(diffuse)
    diffuse = cv2.merge((B, G, R))
    spec = np.uint8(spec * 255)
    diffuse = np.uint8(diffuse * 255)

    if display:
        cv2.imshow("spec", spec)
        cv2.imshow("diffuse", diffuse)
        cv2.waitKey(0)

    return diffuse, spec


def extract_spec_feature(img):
    diffuse, spec = img_split(img)

    # 预处理
    spec_num = np.sum(spec > 0)
    spec_sum = np.sum(spec)
    if spec_num == 0:
        spec_avg = 0
    else:
        spec_avg = spec_sum / spec_num
    spec[spec > 4 * spec_avg] = 0
    spec[spec < 1.5 * spec_avg] = 0
    spec_size = spec.size

    # 特征提取
    spec = np.float32(spec)
    # spec=spec/255
    spec_num = np.sum(spec > 0)

    spec_sum = np.sum(spec)

    if spec_num == 0:
        spec_avg = 0
    else:
        spec_avg = spec_sum / spec_num

    if spec_size == 0:
        spec_per = 0
    else:
        spec_per = spec_num / spec_size

    spec_flat = spec.flatten()
    spec_pixel = np.array(spec_num)
    spec_pixel = spec_flat[spec_flat > 0]
    spec_var = np.var(spec_pixel)

    return [spec_per, spec_avg, spec_var]


def blur_metric(img):
    # 将图像格式转成RGB顺序
    (B, G, R) = cv2.split(img)
    img_rgb = cv2.merge((R, G, B))

    # 转写代码
    img_rgb = np.float64(img_rgb)
    img_rgb = img_rgb / 255

    (w, h, d) = img_rgb.shape
    Hv = np.array((1, 1, 1, 1, 1, 1, 1, 1, 1)) / 9
    Hh = np.transpose(Hv)

    B_Ver = cv2.blur(img_rgb, (1, 30))
    B_Hor = cv2.blur(img_rgb, (30, 1))

    # img_reshape=np.reshape(img_rgb,(w,h*d))
    # D_F_Ver=img_reshape[0:h*d-1]-img_reshape[1:h*d]
    D_F_Ver = abs(img_rgb[:, 0:h - 1] - img_rgb[:, 1:h])
    D_F_Hor = abs(img_rgb[0:w - 1, :] - img_rgb[1:w, :])

    D_F_Ver = np.reshape(D_F_Ver, (w, (h - 1) * d))
    D_F_Hor = np.reshape(D_F_Hor, (w - 1, h * d))

    D_B_Ver = abs(B_Ver[:, 0:h - 1] - B_Ver[:, 1:h])
    D_B_Hor = abs(B_Hor[0:w - 1, :] - B_Hor[1:w, :])
    D_B_Ver = np.reshape(D_B_Ver, (w, (h - 1) * d))
    D_B_Hor = np.reshape(D_B_Hor, (w - 1, h * d))

    T_Ver = D_F_Ver - D_B_Ver
    T_Hor = D_F_Hor - D_B_Hor

    V_Ver = T_Ver
    V_Ver[V_Ver < 0] = 0
    V_Hor = T_Hor
    V_Hor[V_Hor < 0] = 0

    S_D_Ver = np.sum(D_F_Ver[1:w - 1, :])
    S_D_Hor = np.sum(D_F_Hor[1:w - 1, 1:h * d - 1])

    S_V_Ver = np.sum(V_Ver[1:w - 1, :])
    S_V_Hor = np.sum(V_Hor[1:w - 1, 1:h * d - 1])

    blur_F_Ver = (S_D_Ver - S_V_Ver) / S_D_Ver
    blur_F_Hor = (S_D_Hor - S_V_Hor) / S_D_Hor

    blur = max(blur_F_Ver, blur_F_Hor)
    return [blur]


def color_quan(img, display=False):
    (B, G, R) = cv2.split(img)

    for i in np.arange(1, 33):
        B[(((i - 1) * 8) == B) | ((i - 1) * 8 < B) & (B < i * 8)] = i
        G[(((i - 1) * 8) == G) | ((i - 1) * 8 < G) & (G < i * 8)] = i
        R[(((i - 1) * 8) == R) | ((i - 1) * 8 < R) & (R < i * 8)] = i

    img_quan = cv2.merge((B, G, R))
    if display:
        B = B * 8
        G = G * 8
        R = R * 8
        img_show = cv2.merge((B, G, R))
        cv2.imshow("quan", img_show)
        cv2.waitKey(0)
    return img_quan


def extract_color_diversity_features(img):
    img_quan = color_quan(img)
    (B, G, R) = cv2.split(img_quan)
    color_map = R * G * B
    bins = range(32 * 32 * 32 + 2)
    color_bin, b = np.histogram(color_map, bins)
    color_num = np.sum(color_bin > 0)

    color_bin = list(color_bin)
    color_bin.sort(reverse=True)
    color_bin = np.array(color_bin)

    size = 32 * 32 * 32
    color_feature = color_bin[0:100]
    color_feature = list(color_feature)
    color_feature.append(color_num)

    return color_feature


def extract_chrmatic_feature(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    feature = []

    for data in (h, s, v):
        data_sum = np.sum(data[:, :])
        data_max = data.max()
        data_max_num = np.sum(data == data_max)
        data_max_per = data_max_num / data_sum

        data_min = data.min()
        data_min_num = np.sum(data == data_min)
        data_min_per = data_min_num / data_sum

        data_mean = np.mean(data[:, :])
        data_std = np.std(data)

        data = data.flatten()
        data_ske = stats.skew(data)
        feature_one = [data_mean, data_std, data_ske, data_max_per, data_min_per]
        feature = feature + feature_one
    return feature


def feature_extract(img, isface=False):
    # 判断是否正确读到图像
    if img is None:
        print("img is None")
        os._exit(0)

    # 人脸检测
    if isface:
        face_roi = img
        # if face_roi.shape[0] < 270:
        #     face_roi = cv2.resize(face_roi, (280, 280))
    else:
        # 人脸检测
        face_roi = face_detector.face_detect(img)
        if face_roi is None:
            face_roi = img

    spec_feature = extract_spec_feature(face_roi)
    blur1 = blur_metric(face_roi)
    blur2 = blur_detect2(face_roi)
    chrmatic_feature = extract_chrmatic_feature(face_roi)
    color_diversity_feature = extract_color_diversity_features(face_roi)
    feature = spec_feature + blur1 + blur2 + chrmatic_feature + color_diversity_feature
    feature_len = len(feature)

    return feature


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
    if not os.path.exists('../' + save_path):
        os.makedirs('../' + save_path)
    with open('../' + save_path + '/IDA_train_feature', 'wb') as lf:
        pickle.dump(train_feature, lf)
    with open('../' + save_path + '/IDA_train_label', 'wb') as ll:
        pickle.dump(train_label, ll)

    with open('../' + save_path + '/IDA_test_feature', 'wb') as lf:
        pickle.dump(test_feature, lf)
    with open('../'+save_path + '/IDA_test_label', 'wb') as ll:
        pickle.dump(test_label, ll)


def train_and_test(feature_path='.'):
    # 读取
    # 存储
    with open('../' + feature_path + '/IDA_train_feature', 'rb') as lf:
        feature_train = pickle.load(lf)
    with open('../' + feature_path + '/IDA_train_label', 'rb') as ll:
        label_train = pickle.load(ll)

    with open('../' + feature_path + '/IDA_test_feature', 'rb') as lf:
        feature_test = pickle.load(lf)
    with open('../' + feature_path + '/IDA_test_label', 'rb') as ll:
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

    where_are_nan = np.isnan(feature_train)
    feature_train[where_are_nan] = 0

    where_are_nan = np.isnan(feature_test)
    feature_test[where_are_nan] = 0

    shuffle_ix = np.random.permutation(np.arange(len(feature_train)))
    feature_train = feature_train[shuffle_ix]
    label_train = label_train[shuffle_ix]

    # 训练
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
    # s=pickle.dumps(model)
    # f=open('SVM.model','wb+')
    # f.write(s)
    # f.close()
    with open('../' + feature_path + '/IDA_model', 'wb') as lm:
        pickle.dump(model, lm)
    return model


def IDA_model_test():
    with open('/home/shicaiwei/information_form_idcard/face_detection/reproduce/IDA/intra_testing/IDA_model',
              'rb') as lm:
        IDA_model = pickle.load(lm)

    time_begin = datetime.datetime.now()
    img_path = '/home/shicaiwei/data/liveness_data/intra_testing/test/living'
    label=1
    isface = False
    file_list = get_file_list(img_path)
    count = 1
    true_num = 1
    for file in file_list:
        img = cv2.imread(file)
        if img is None:
            continue
        if not isface:
            img = cv2.resize(img, (480, 640))
        img_feature = feature_extract(img, isface=isface)
        if img_feature is None:
            continue

        img_feature = np.array(img_feature)
        where_are_nan = np.isnan(img_feature)
        img_feature[where_are_nan] = 0
        img_feature = list(img_feature)

        try:
            result = IDA_model.predict([img_feature])
            print(result)
            count += 1
            if result == label:
                true_num += 1
            else:
                print(file)
            # if count > 180:
            #     break
        except Exception as e:
            pass

    print(count, true_num, true_num / count)

    time_end = datetime.datetime.now()
    time_all = time_end - time_begin
    print("time_all", time_all.total_seconds())


def IDA_feature_test():
    '''
    IDA feature的测试
    :return:
    '''
    img = cv2.imread("/home/shicaiwei/information_form_idcard/face_detection/data_set_make/ipad2/360.jpg")
    # img = cv2.imread("light.png")
    # img_split(img)
    # blur_metric(img)
    extract_spec_feature(img)
    # color_diversity_features(img)
    # extract_chrmatic_feature(img)
    feature = feature_extract(img)
    # print(1)


def IDA_mdoel_train(train_dir, test_dir, dataset_name=''):
    save_path = dataset_name
    isface = False
    data_make(train_dir=train_dir, test_dir=test_dir, save_path=save_path, isface=isface)
    train_and_test(feature_path=save_path)


if __name__ == '__main__':
    train_dir = "/home/shicaiwei/data/liveness_data/intra_testing_face/train"
    test_dir = "/home/shicaiwei/data/liveness_data/intra_testing_face/val"
    save_path = 'intra_testing'
    space = 'gray'
    
    # 输入的图像是否是人脸图像,是的话,则不会再进行人脸检测,不是则会.
    isface = True
    data_make(train_dir=train_dir, test_dir=test_dir, save_path=save_path, isface=isface)
    train_and_test(feature_path=save_path)
    IDA_model_test()
