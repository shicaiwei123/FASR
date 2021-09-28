import os
import cv2
import sys

sys.path.append('..')
from lib.processing_utils import LandmarksDetection,frame_to_face


def bottom_generate(face_dir, bottom_dir, save_mode='.bmp', display=False):
    '''
    输入人脸图像，获取下半部分人脸图像
    :param face_dir:
    :param bottom_dir:
    :param save_mode:
    :return:
    '''

    count = 1
    landmarks_detector = LandmarksDetection()
    for root, dirs, files in os.walk(face_dir):

        # 当子目录为空的时候，root就是不包含子文件夹的文件夹
        if dirs == []:
            files = sorted(files)
            for file_name in files:
                file_path = os.path.join(root, file_name)
                img = cv2.imread(file_path)
                print(img.shape)
                if img is None:
                    print("if img is None:")
                    continue

                # 获取存储路径
                face_dir_split = face_dir.split('/')
                file_path_split = file_path.split('/')
                file_path_split.pop()  # 去掉文件名
                sub_split = [item for item in file_path_split if item not in face_dir_split]
                save_dir = bottom_dir
                for item in sub_split:
                    save_dir = os.path.join(save_dir, item)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 截取bottom
                count += 1

                landmarks = landmarks_detector.landmarks_detect(img)
                # 人脸裁剪
                key = landmarks
                left = key[0][0]  # 第一个特征点的纵坐标,也就是宽的方向
                right = key[16][0]  # 第17个特征点的纵坐标,也就是宽方形
                top = key[30][1]
                bottom = key[8][1]  # 第9个点的横坐标
                if left < 0:
                    left = 0
                img_roi = img[top:bottom, left:right]
                try:
                    if img_roi is None:
                        continue
                    img_roi = cv2.resize(img_roi, (224, 112))
                    if display:
                        cv2.imshow("roi", img_roi)
                        cv2.waitKey(0)

                    # 存储
                    save_path = os.path.join(save_dir, file_name.split('.')[0] + save_mode)
                    cv2.imwrite(save_path, img_roi)

                    print(count)
                except Exception as e:
                    print(e)


if __name__ == '__main__':


    face_dir = "/home/shicaiwei/data/liveness_data/intra_testing_face"
    bottom_dir = "/home/shicaiwei/data/liveness_data/intra_testing_bottom"

    bottom_generate(face_dir=face_dir, bottom_dir=bottom_dir, display=False)
