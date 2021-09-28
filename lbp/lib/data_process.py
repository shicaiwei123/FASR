import sys

sys.path.append('..')
import cv2
import os
import numpy as np
import sys
from lib.processing_utils import read_txt, get_file_list
import shutil
import datetime
import time

'''
数据和数据集的一些预处理操作,包括但不限于:从视频中获取图像,人脸检测,利用原始数据集,生成满足条件的数据集.
'''


def video_to_frames(pathIn='',
                    pathOut='',
                    extract_time_interval=-1,
                    only_output_video_info=False,
                    extract_time_points=None,
                    initial_extract_time=0,
                    end_extract_time=None,
                    output_prefix='frame',
                    jpg_quality=100,
                    isColor=True):
    '''
    pathIn：视频的路径，比如：F:\python_tutorials\test.mp4
    pathOut：设定提取的图片保存在哪个文件夹下，比如：F:\python_tutorials\frames1\。如果该文件夹不存在，函数将自动创建它
    only_output_video_info：如果为True，只输出视频信息（长度、帧数和帧率），不提取图片
    extract_time_points：提取的时间点，单位为秒，为元组数据，比如，(2, 3, 5)表示只提取视频第2秒， 第3秒，第5秒图片
    initial_extract_time：提取的起始时刻，单位为秒，默认为0（即从视频最开始提取）
    end_extract_time：提取的终止时刻，单位为秒，默认为None（即视频终点）
    extract_time_interval：提取的时间间隔，单位为秒，默认为-1（即输出时间范围内的所有帧）
    output_prefix：图片的前缀名，默认为frame，图片的名称将为frame_000001.jpg、frame_000002.jpg、frame_000003.jpg......
    jpg_quality：设置图片质量，范围为0到100，默认为100（质量最佳）
    isColor：如果为False，输出的将是黑白图片
    '''

    cap = cv2.VideoCapture(pathIn)  ##打开视频文件
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  ##视频的帧数
    fps = cap.get(cv2.CAP_PROP_FPS)  ##视频的帧率
    dur = n_frames / fps  ##视频的时间

    ##如果only_output_video_info=True, 只输出视频信息，不提取图片
    if only_output_video_info:
        print('only output the video information (without extract frames)::::::')
        print("Duration of the video: {} seconds".format(dur))
        print("Number of frames: {}".format(n_frames))
        print("Frames per second (FPS): {}".format(fps))

        ##提取特定时间点图片
    elif extract_time_points is not None:
        if max(extract_time_points) > dur:  ##判断时间点是否符合要求
            raise NameError('the max time point is larger than the video duration....')
        try:
            os.mkdir(pathOut)
        except OSError:
            pass
        success = True
        count = 0
        while success and count < len(extract_time_points):
            cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * extract_time_points[count]))
            success, image = cap.read()
            if success:
                if not isColor:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  ##转化为黑白图片
                print('Write a new frame: {}, {}th'.format(success, count + 1))
                cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.jpg".format(output_prefix, count + 1)), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                count = count + 1

    else:
        ##判断起始时间、终止时间参数是否符合要求
        if initial_extract_time > dur:
            raise NameError('initial extract time is larger than the video duration....')
        if end_extract_time is not None:
            if end_extract_time > dur:
                raise NameError('end extract time is larger than the video duration....')
            if initial_extract_time > end_extract_time:
                raise NameError('end extract time is less than the initial extract time....')

        ##时间范围内的每帧图片都输出
        if extract_time_interval == -1:
            if initial_extract_time > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time))
            try:
                os.mkdir(pathOut)
            except OSError:
                pass
            print('Converting a video into frames......')
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) * fps + 1
                success = True
                count = 0
                while success and count < N:
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}/{}'.format(success, count + 1, n_frames))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1
            else:
                success = True
                count = 0
                while success:
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}/{}'.format(success, count + 1, n_frames))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1

        ##判断提取时间间隔设置是否符合要求
        elif extract_time_interval > 0 and extract_time_interval < 1 / fps:
            raise NameError('extract_time_interval is less than the frame time interval....')
        elif extract_time_interval > (n_frames / fps):
            raise NameError('extract_time_interval is larger than the duration of the video....')

        ##时间范围内每隔一段时间输出一张图片
        else:
            try:
                os.mkdir(pathOut)
            except OSError:
                pass
            print('Converting a video into frames......')
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) / extract_time_interval + 1
                success = True
                count = 0
                while success and count < N:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}th'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1
            else:
                success = True
                count = 0
                while success:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not isColor:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('Write a new frame: {}, {}th'.format(success, count + 1))
                        cv2.imwrite(os.path.join(pathOut, "{}_{:06d}.bmp".format(output_prefix, count + 1)), image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])  # save frame as JPEG file
                        count = count + 1


class FaceDection(object):
    def __init__(self, model_name):
        if model_name == "CAFFE":
            modelFile = "lib/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "lib/deploy.prototxt"
            net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        else:
            modelFile = "../lib/opencv_face_detector_uint8.pb"
            configFile = "../lib/opencv_face_detector.pbtxt"
            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        self.model = net
        self.conf_threshold = 0.7

    def face_detect(self, img, display=False):
        '''
        输入人脸,返回人脸照片
        :param img:
        :param display:
        :return:
        '''
        frameOpencvDnn = img.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        img_mean = np.mean(frameOpencvDnn, (0, 1))

        # 数据预处理
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), img_mean, False, False)

        self.model.setInput(blob)
        detections = self.model.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        if display:
            cv2.imshow("frame", frameOpencvDnn)
            cv2.waitKey(0)

        # 如果检测到多个人脸,取最大的人脸
        if len(bboxes) > 1:
            bbox_max = [0, 0, 0, 0]
            bbox_max_len = bbox_max[2] - bbox_max[0]
            for bbox in bboxes:
                if (bbox[2] - bbox[0]) > bbox_max_len:
                    bbox_max = bbox
                    bbox_max_len = bbox_max[2] - bbox_max[0]

            face_img = img[bbox_max[1]:bbox_max[3], bbox_max[0]:bbox_max[2]]
        elif len(bboxes) == 0:
            '''检测不到人脸'''
            return None
        else:
            bbox_max = bboxes[0]
            face_img = img[bbox_max[1]:bbox_max[3], bbox_max[0]:bbox_max[2]]

        return face_img


def nuaa_dataset_generate():
    '''
    将NUAA数据集,转变成我们需要的文件夹分级格式
    :return:
    '''

    # 初始化
    NUAA_raw_path = "/home/bbb/shicaiwei/data/liveness_data/NUAA/raw"
    client_train_raw = NUAA_raw_path + '/client_train_raw.txt'
    client_test_raw = NUAA_raw_path + '/client_test_raw.txt'
    imposter_train_raw = NUAA_raw_path + '/imposter_train_raw.txt'
    imposter_test_raw = NUAA_raw_path + '/imposter_test_raw.txt'

    # 生成文件夹
    train_spoofing_path = 'NUAA/train/spoofing'
    if not os.path.exists(train_spoofing_path):
        os.makedirs(train_spoofing_path)
    train_living_path = 'NUAA/train/living'
    if not os.path.exists(train_living_path):
        os.makedirs(train_living_path)

    test_spoofing_path = 'NUAA/test/spoofing'
    if not os.path.exists(test_spoofing_path):
        os.makedirs(test_spoofing_path)
    test_living_path = 'NUAA/test/living'
    if not os.path.exists(test_living_path):
        os.makedirs(test_living_path)

    # 数据准备
    client_train_list = read_txt(client_train_raw)
    for relative_path in client_train_list:
        img_path = NUAA_raw_path + '/ClientRaw/' + relative_path
        shutil.copy(img_path, train_living_path)

    client_test_list = read_txt(client_test_raw)
    for relative_path in client_test_list:
        img_path = NUAA_raw_path + '/ClientRaw/' + relative_path
        shutil.copy(img_path, test_living_path)

    imposter_train_list = read_txt(imposter_train_raw)
    for relative_path in imposter_train_list:
        img_path = NUAA_raw_path + '/ImposterRaw/' + relative_path
        shutil.copy(img_path, train_spoofing_path)

    imposter_test_list = read_txt(imposter_test_raw)
    for relative_path in imposter_test_list:
        img_path = NUAA_raw_path + '/ImposterRaw/' + relative_path
        shutil.copy(img_path, test_spoofing_path)


def multi_video_to_frame(video_path_list, save_dir):
    '''
    将一个文件夹下的多个视频,全部转成帧图像
    :param videos_path:
    :param save_path:
    :return:
    '''

    count_all = 1
    for video_path in video_path_list:
        cap = cv2.VideoCapture(video_path)  ##打开视频文件
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  ##视频的帧数
        success = True
        count_single = 0
        while success and count_single < n_frames:
            success, image = cap.read()
            if success:
                img_name = str(count_all) + '.bmp'

                save_path = os.path.join(save_dir, img_name)
                cv2.imwrite(save_path, image)

                count_single += 1
                count_all += 1
    print("count_all", count_all)


def replayattack_dataset_generate():
    '''
    将原始的replayattack 数据集转成我们需要的格式
    :return:
    '''

    # 初始化
    replayattack_path = "/home/bbb/shicaiwei/data/liveness_data/replayattack_video"
    train_real_path = replayattack_path + '/train/real'
    train_attack_path = replayattack_path + '/train/attack'
    test_real_path = replayattack_path + '/test/real'
    test_attack_path = replayattack_path + '/test/attack'

    # 生成文件夹
    train_spoofing_path = 'replayattack/train/spoofing'
    if not os.path.exists(train_spoofing_path):
        os.makedirs(train_spoofing_path)
    train_living_path = 'replayattack/train/living'
    if not os.path.exists(train_living_path):
        os.makedirs(train_living_path)

    test_spoofing_path = 'replayattack/test/spoofing'
    if not os.path.exists(test_spoofing_path):
        os.makedirs(test_spoofing_path)
    test_living_path = 'replayattack/test/living'
    if not os.path.exists(test_living_path):
        os.makedirs(test_living_path)

    # 数据集生成
    train_real_path_list = get_file_list(train_real_path)
    multi_video_to_frame(train_real_path_list, train_living_path)

    # attack
    train_attack_path_list = get_file_list(train_attack_path)
    replay_list = []
    print_list = []
    for path in train_attack_path_list:
        path_split = path.split('/')
        video_name = path_split[-1]
        video_name_split = video_name.split('_')
        if video_name_split[1] == "highdef" or video_name_split[1] == "print":
            print_list.append(path)
        else:
            replay_list.append(path)
    train_replay_path = train_spoofing_path + '/replay'
    if not os.path.exists(train_replay_path):
        os.makedirs(train_replay_path)
    train_print_path = train_spoofing_path + '/print'
    if not os.path.exists(train_print_path):
        os.makedirs(train_print_path)
    multi_video_to_frame(replay_list, train_replay_path)
    multi_video_to_frame(print_list, train_print_path)

    # real
    test_real_path_list = get_file_list(test_real_path)
    multi_video_to_frame(test_real_path_list, test_living_path)

    # attack
    test_attack_path_list = get_file_list(test_attack_path)
    replay_list = []
    print_list = []
    for path in test_attack_path_list:
        path_split = path.split('/')
        video_name = path_split[-1]
        video_name_split = video_name.split('_')
        if video_name_split[1] == "highdef" or video_name_split[1] == "print":
            print_list.append(path)
        else:
            replay_list.append(path)
    test_replay_path = test_spoofing_path + '/replay'
    if not os.path.exists(test_replay_path):
        os.makedirs(test_replay_path)
    test_print_path = test_spoofing_path + '/print'
    if not os.path.exists(test_print_path):
        os.makedirs(test_print_path)
    multi_video_to_frame(replay_list, test_replay_path)
    multi_video_to_frame(print_list, test_print_path)


def casia_fasd_generate():
    '''
    将原始的casia_fasd 数据集转成我们需要的格式
    :return:
    '''

    # 初始化
    living_index_list = ['1', '2', 'HR1']
    print_index_list = ['3', '4', '5', '6', 'HR2', 'HR3']
    replay_index_list = ['7', '8', 'HR4']
    casia_fasd_dir = "/home/bbb/shicaiwei/data/liveness_data/CASIA-FASD"
    train_dir = casia_fasd_dir + '/train_release'
    test_dir = casia_fasd_dir + '/test_release'

    # 生成文件夹
    train_spoofing_path = 'CASIA-FASD/train/spoofing'
    if not os.path.exists(train_spoofing_path):
        os.makedirs(train_spoofing_path)
    train_living_path = 'CASIA-FASD/train/living'
    if not os.path.exists(train_living_path):
        os.makedirs(train_living_path)

    test_spoofing_path = 'CASIA-FASD/test/spoofing'
    if not os.path.exists(test_spoofing_path):
        os.makedirs(test_spoofing_path)
    test_living_path = 'CASIA-FASD/test/living'
    if not os.path.exists(test_living_path):
        os.makedirs(test_living_path)

    # 数据处理

    # 训练
    train_dir_floder = os.listdir(train_dir)
    train_living_video_list = []
    train_print_video_list = []
    train_replay_video_list = []

    # 获取路径list
    for floder_index in train_dir_floder:
        video_dir = os.path.join(train_dir, floder_index)
        for video_index in living_index_list:
            video_name = video_index + '.avi'
            video_path = os.path.join(video_dir, video_name)
            train_living_video_list.append(video_path)

        for video_index in print_index_list:
            video_name = video_index + '.avi'
            video_path = os.path.join(video_dir, video_name)
            train_print_video_list.append(video_path)

        for video_index in replay_index_list:
            video_name = video_index + '.avi'
            video_path = os.path.join(video_dir, video_name)
            train_replay_video_list.append(video_path)

    multi_video_to_frame(train_living_video_list, train_living_path)

    save_path = train_spoofing_path + '/print'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    multi_video_to_frame(train_print_video_list, save_path)

    save_path = train_spoofing_path + '/replay'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    multi_video_to_frame(train_replay_video_list, save_path)

    # 测试
    test_dir_floder = os.listdir(train_dir)
    test_living_video_list = []
    test_print_video_list = []
    test_replay_video_list = []

    # 获取路径list
    for floder_index in test_dir_floder:
        video_dir = os.path.join(test_dir, floder_index)
        for video_index in living_index_list:
            video_name = video_index + '.avi'
            video_path = os.path.join(video_dir, video_name)
            test_living_video_list.append(video_path)

        for video_index in print_index_list:
            video_name = video_index + '.avi'
            video_path = os.path.join(video_dir, video_name)
            test_print_video_list.append(video_path)

        for video_index in replay_index_list:
            video_name = video_index + '.avi'
            video_path = os.path.join(video_dir, video_name)
            test_replay_video_list.append(video_path)

    multi_video_to_frame(test_living_video_list, test_living_path)

    save_path = test_spoofing_path + '/print'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    multi_video_to_frame(test_print_video_list, save_path)

    save_path = test_spoofing_path + '/replay'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    multi_video_to_frame(test_replay_video_list, save_path)


def msu_mfsd_generate():
    '''
    转换msu mfsd 数据集到我们需要的格式
    :return:
    '''
    # 初始化
    msu_path = "/home/bbb/shicaiwei/data/liveness_data/msu_mfsd"
    train_sub_txt = msu_path + '/train_sub_list.txt'
    test_sub_txt = msu_path + '/test_sub_list.txt'

    # 生成文件夹
    train_spoofing_path = 'MSU-MFSD/train/spoofing'
    if not os.path.exists(train_spoofing_path):
        os.makedirs(train_spoofing_path)
    train_living_path = 'MSU-MFSD/train/living'
    if not os.path.exists(train_living_path):
        os.makedirs(train_living_path)

    test_spoofing_path = 'MSU-MFSD/test/spoofing'
    if not os.path.exists(test_spoofing_path):
        os.makedirs(test_spoofing_path)
    test_living_path = 'MSU-MFSD/test/living'
    if not os.path.exists(test_living_path):
        os.makedirs(test_living_path)

    # 数据集生成

    file_name_list = os.listdir(msu_path)

    for file_name in file_name_list:

        if 'mov' in file_name or 'mp4' in file_name:
            # 转avi
            file_path = os.path.join(msu_path, file_name)
            file_path_split = file_path.split('.')
            file_path_split[-1] = 'avi'
            file_path_new = '.'.join(file_path_split)
            os.rename(file_path, file_path_new)

    # train
    train_index_list = read_txt(train_sub_txt)
    real_list = []
    replay_list = []
    print_list = []
    # 遍历index 获取用来训练的数据
    for index in train_index_list:
        index = index + "_"
        for file_name in file_name_list:

            if 'avi' in file_name:

                if index in file_name:
                    video_path = os.path.join(msu_path, file_name)
                    file_name_split = file_name.split('_')
                    if file_name_split[0] == 'real':
                        real_list.append(video_path)
                    else:
                        if file_name_split[4] == "printed":
                            print_list.append(video_path)
                        else:
                            replay_list.append(video_path)

    multi_video_to_frame(real_list, train_living_path)

    train_replay_path = train_spoofing_path + '/replay'
    if not os.path.exists(train_replay_path):
        os.makedirs(train_replay_path)
    train_print_path = train_spoofing_path + '/print'
    if not os.path.exists(train_print_path):
        os.makedirs(train_print_path)
    multi_video_to_frame(replay_list, train_replay_path)
    multi_video_to_frame(print_list, train_print_path)

    # test
    test_index_list = read_txt(test_sub_txt)
    real_list = []
    replay_list = []
    print_list = []
    # 遍历index 获取用来训练的数据
    for index in test_index_list:
        for file_name in file_name_list:
            if 'avi' in file_name:
                if index in file_name:
                    video_path = os.path.join(msu_path, file_name)
                    file_name_split = file_name.split('_')
                    if file_name_split[0] == 'real':
                        real_list.append(video_path)
                    else:
                        if file_name_split[4] == "printed":
                            print_list.append(video_path)
                        else:
                            replay_list.append(video_path)

    multi_video_to_frame(real_list, test_living_path)

    test_replay_path = test_spoofing_path + '/replay'
    if not os.path.exists(test_replay_path):
        os.makedirs(test_replay_path)
    test_print_path = test_spoofing_path + '/print'
    if not os.path.exists(test_print_path):
        os.makedirs(test_print_path)
    multi_video_to_frame(replay_list, test_replay_path)
    multi_video_to_frame(print_list, test_print_path)


def oulu_generate():
    '''
    将oulu 数据集转成我们需要的格式
    :return:
    '''
    # 初始化
    msu_path = "/home/bbb/shicaiwei/data/liveness_data/OULU"
    train_dir = msu_path + '/Train_files'
    test_dir = msu_path + '/Train_files'

    # 生成文件夹
    train_spoofing_path = 'OULU/train/spoofing'
    if not os.path.exists(train_spoofing_path):
        os.makedirs(train_spoofing_path)
    train_living_path = 'OULU/train/living'
    if not os.path.exists(train_living_path):
        os.makedirs(train_living_path)

    test_spoofing_path = 'OULU/test/spoofing'
    if not os.path.exists(test_spoofing_path):
        os.makedirs(test_spoofing_path)
    test_living_path = 'OULU/test/living'
    if not os.path.exists(test_living_path):
        os.makedirs(test_living_path)

    # 数据处理

    train_file_list = os.listdir(train_dir)
    real_list = []
    replay_list = []
    print_list = []

    for file in train_file_list:
        if 'txt' in file:
            continue
        file_split = file.split('_')
        video_path = os.path.join(train_dir, file)
        if file_split[3] == '1':
            real_list.append(video_path)
        else:
            if file_split[3] == '2' or file_split[3] == '3':
                print_list.append(video_path)
            else:
                replay_list.append(video_path)

    multi_video_to_frame(real_list, train_living_path)

    train_replay_path = train_spoofing_path + '/replay'
    if not os.path.exists(train_replay_path):
        os.makedirs(train_replay_path)
    train_print_path = train_spoofing_path + '/print'
    if not os.path.exists(train_print_path):
        os.makedirs(train_print_path)
    multi_video_to_frame(replay_list, train_replay_path)
    multi_video_to_frame(print_list, train_print_path)

    # test
    test_file_list = os.listdir(test_dir)
    real_list = []
    replay_list = []
    print_list = []

    for file in test_file_list:
        if 'txt' in file:
            continue
        file_split = file.split('_')
        video_path = os.path.join(test_dir, file)
        if file_split[3] == '1':
            real_list.append(video_path)
        else:
            if file_split[3] == '2' or file_split[3] == '3':
                print_list.append(video_path)
            else:
                replay_list.append(video_path)

    # 存储
    multi_video_to_frame(real_list, test_living_path)

    test_replay_path = test_spoofing_path + '/replay'
    if not os.path.exists(test_replay_path):
        os.makedirs(test_replay_path)
    test_print_path = test_spoofing_path + '/print'
    if not os.path.exists(test_print_path):
        os.makedirs(test_print_path)
    multi_video_to_frame(replay_list, test_replay_path)
    multi_video_to_frame(print_list, test_print_path)


def frame_to_face(frame_dir, face_dir):
    '''
    对生成的数据集,进行人脸检测,获取人脸,加速后面的训练
    :return:
    '''

    # 初始化

    frame_train_dir = os.path.join(frame_dir, 'train')
    frame_train_living_dir = os.path.join(frame_train_dir, 'living')
    frame_train_spoofing_dir = os.path.join(frame_train_dir, 'spoofing')
    frame_test_dir = os.path.join(frame_dir, 'test')
    frame_test_living_dir = os.path.join(frame_test_dir, 'living')
    frame_test_spoofing_dir = os.path.join(frame_test_dir, 'spoofing')

    face_train_dir = os.path.join(face_dir, 'train')
    face_train_living_dir = os.path.join(face_train_dir, 'living')
    face_train_spoofing_dir = os.path.join(face_train_dir, 'spoofing')
    face_test_dir = os.path.join(face_dir, 'test')
    face_test_living_dir = os.path.join(face_test_dir, 'living')
    face_test_spoofing_dir = os.path.join(face_test_dir, 'spoofing')

    face_detection = FaceDection(model_name='TF')

    # 生成文件夹
    if not os.path.exists(face_train_living_dir):
        os.makedirs(face_train_living_dir)
    if not os.path.exists(face_train_spoofing_dir):
        os.makedirs(face_train_spoofing_dir)
    if not os.path.exists(face_test_living_dir):
        os.makedirs(face_test_living_dir)
    if not os.path.exists(face_test_spoofing_dir):
        os.makedirs(face_test_spoofing_dir)

    # 人脸检测并保存
    file_path_list = get_file_list(frame_train_living_dir)
    file_len = len(file_path_list)
    file_count = 1
    for file_path in file_path_list:
        print("{}/{}".format(file_count, file_len))
        img_name = file_path.split('/')[-1]
        img = cv2.imread(file_path)
        face_img = face_detection.face_detect(img)
        if face_img is None:
            continue
        save_path = os.path.join(face_train_living_dir, img_name)
        cv2.imwrite(save_path, face_img)
        file_count += 1

    file_path_list = get_file_list(frame_test_living_dir)
    file_len = len(file_path_list)
    file_count = 1
    for file_path in file_path_list:
        print("{}/{}".format(file_count, file_len))
        img_name = file_path.split('/')[-1]
        img = cv2.imread(file_path)
        face_img = face_detection.face_detect(img)
        if face_img is None:
            continue
        save_path = os.path.join(face_test_living_dir, img_name)
        cv2.imwrite(save_path, face_img)
        file_count += 1

    file_path_list = get_file_list(frame_train_spoofing_dir)
    file_len = len(file_path_list)
    file_count = 1
    for file_path in file_path_list:
        print("{}/{}".format(file_count, file_len))
        img_name = file_path.split('/')[-1]
        img = cv2.imread(file_path)
        face_img = face_detection.face_detect(img)
        if face_img is None:
            continue
        save_path = os.path.join(face_train_spoofing_dir, img_name)
        cv2.imwrite(save_path, face_img)
        file_count += 1

    file_path_list = get_file_list(frame_test_spoofing_dir)
    file_len = len(file_path_list)
    file_count = 1
    for file_path in file_path_list:
        print("{}/{}".format(file_count, file_len))
        img_name = file_path.split('/')[-1]
        img = cv2.imread(file_path)
        face_img = face_detection.face_detect(img)
        if face_img is None:
            continue
        save_path = os.path.join(face_test_spoofing_dir, img_name)
        cv2.imwrite(save_path, face_img)
        file_count += 1


if __name__ == '__main__':
    # video_to_frame test
    # video_path = "/home/shicaiwei/data/liveness_data/replay_attack/hand/attack_highdef_client110_session01_highdef_photo_adverse.mov"
    # save_path = "/home/shicaiwei/data/liveness_data/replay_attack/hang_img"
    # video_to_frames(pathIn=video_path, pathOut=save_path)

    # face_deetct test

    # time_begin = time.time()
    # img_path = '/home/shicaiwei/data/liveness_data/CMAS/living_cross_test/2/0.jpg'
    # img = cv2.imread(img_path)
    # face_img = face_detect(img)
    # time_end = time.time()
    # detected_face_time = time_end - time_begin
    # print("face_detected time:", detected_face_time)
    # cv2.imshow("face", face_img)
    # cv2.waitKey(0)

    # NUAA dataset
    # nuaa_dataset_generate()

    # replay
    # replayattack_dataset_generate()

    # # caisa
    # casia_fasd_generate()

    # msu_mfsd
    msu_mfsd_generate()

    ##oulu
    # oulu_generate()

    # # frame to face
    # frame_dir = "/home/bbb/shicaiwei/data/liveness_data/replayattack_video/replayattack"
    # face_dir = "/home/bbb/shicaiwei/data/liveness_data/replayattack_video/replayattack_face"
    # frame_to_face(frame_dir=frame_dir, face_dir=face_dir)
