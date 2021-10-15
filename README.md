# FASR
Code and Review (survey) for some classical and SOTA face anti-spoofing detection (E.g. LBP, ColorLBP, IDA, patch-based-cnn,CDCN)

## DataSet:
- Single Modality:
    - Replay-Attack
    - CASIA-FASD
    - MSU-MFSD
    - OULU-NPU
    - SIW
    - SIW-M
    - CelebA
- Multimodality
    - CASIA-SURF
    - CeFA-SURF
    - HQ-WMCA
## Proprocessing
- video to frame
    ```python
    from lib.processing_utils import video_to_frames
    video_to_frames(video_path,output_path)
    ```
- frame to face
    ```python
    from lib.processing_utils import frame_to_face
    frame_to_face(frame_path,output_path)
    ```
- extract face with landmarks
    

    
