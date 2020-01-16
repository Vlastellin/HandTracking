# HandTracking

We try to create a program, which can track hand movement
Link to a marked dataset with 9 hand position classes - https://drive.google.com/drive/u/1/folders/1nMqigKFVqS4SWyuLEa5u748ufrmHy8--

The first step: You must download the 3D detector from this source: https://github.com/lmb-freiburg/hand3d.
To start a project: install tensorflow 1.14.0, scipy 1.4.1, opencv-python, imageio, numpy, pickle, matplotlib, tqdm.

Secondly: copy and paste all files from our git master branch to 3D  detector project.

To launch this project, you need to initialize command prompt window in root directory and type in "python parser.py --video={video_path}" where video_path is a name or location of a video you want to process. Then, when programm is finished, result video will appear as an "output.mp4" file in root directory.

utils/find_angle.py - a utility that allows you to describe the position of the hand in space relative to three coordinate axes. To use you need to import the method description_of_hand_position(Input: keypoint_coord3d_v -   [1, 21, 3] tf.float32 tensor, Normalizable 3D coordinates of  keypoints).

Note: in other branches (non master) located other trys of solving our hand tracking problem
    handkeydetection branch:
                            hand_key_detect.py - script for finding 2d coordinates of hands key points
                            display_data_in_graph.py - script for displaying experimental data on 2D graphs
                            data.pickle - list of 2d coorinates of key points of collected experimental data
                            
                            