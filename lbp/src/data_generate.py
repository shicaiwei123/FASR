
import sys
sys.path.append('..')

from lib.processing_utils import frame_to_face

if __name__ == '__main__':
    frame_dir = "/home/shicaiwei/data/liveness_data/cross_replayed"
    face_dir = "/home/shicaiwei/data/liveness_data/cross_replayed_face_normal"
    frame_to_face(frame_dir=frame_dir, face_dir=face_dir, model_name='cv2', normal_size=(480, 640))
