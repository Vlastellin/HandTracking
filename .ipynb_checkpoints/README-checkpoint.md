# HandTracking
We try to create a program, which can track hand movement in real time
Link to a marked dataset with 9 hand position classes - https://drive.google.com/drive/u/1/folders/1nMqigKFVqS4SWyuLEa5u748ufrmHy8--

utils/find_angle.py - a utility that allows you to describe the position of the hand in space relative to three coordinate axes. For use you need  import the method description_of_hand_position(Input: keypoint_coord3d_v -   [1, 21, 3] tf.float32 tensor, Normalizable 3D coordinates of  keypoints).
