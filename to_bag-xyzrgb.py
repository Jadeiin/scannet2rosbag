# -*- coding:UTF-8 -*-
import cv2
import time, sys, os, struct
from glob import glob
from ros import rosbag
import roslib
import rospy

roslib.load_manifest("sensor_msgs")
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcl2
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Transform
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from pyquaternion import Quaternion
import tf
from PIL import ImageFile
from PIL import Image as ImagePIL

all_num = 5578

def ReadImages(filename):
    all = []
    for i in range(all_num):
        all.append(filename + str(i) + ".png")
    return all


def ReadRGBImages(filename):
    all = []
    for i in range(all_num):
        all.append(filename + str(i) + ".jpg")
    return all


def ReadPose(filename):
    posedata = []
    for i in range(all_num):
        data = np.loadtxt(filename + str(i) + ".txt")
        posedata.append(data)
    return posedata


def depth2xyz(depth_map, fx, fy, cx, cy, flatten=False, depth_scale=1000):
    h, w = np.mgrid[0 : depth_map.shape[0], 0 : depth_map.shape[1]]
    z = depth_map / depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    xyz = (
        np.dstack((x, y, z))
        if flatten == False
        else np.dstack((x, y, z)).reshape(-1, 3)
    )
    # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz


def depth2xyzrgb(depth_map, B, G, R, fx, fy, cx, cy, flatten=False, depth_scale=1000):
    h, w = np.mgrid[0 : depth_map.shape[0], 0 : depth_map.shape[1]]
    z = depth_map / depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy

    # 1 channel, with the float in the channel reinterpreted as 3 single-byte values with ranges from 0 to 255 like 0xff0000
    rgb = R.astype(np.uint32) << 16 | G.astype(np.uint32) << 8 | B.astype(np.uint32)

    xyzrgb = (
        np.dstack((x, y, z, rgb))
        if flatten == False
        else np.dstack((x, y, z, rgb)).reshape(-1, 4)
    )
    return xyzrgb


# def depth2xyzrgbl(
#     depth_map, B, G, R, semantic_image, fx, fy, cx, cy, flatten=False, depth_scale=1000
# ):
#     h, w = np.mgrid[0 : depth_map.shape[0], 0 : depth_map.shape[1]]
#     z = depth_map / depth_scale
#     x = (w - cx) * z / fx
#     y = (h - cy) * z / fy
#     # r = R
#     # b = B
#     # g =  G
#     # print(r[100,:])
#     a = np.ones((480, 640))
#     for i in range(0, 480):
#         for j in range(0, 640):
#             a[i][j] = 255
#     r = a - R
#     b = a - B
#     g = a - G

#     l = semantic_image
#     xyzrgbl = (
#         np.dstack((x, y, z, r, g, b, l))
#         if flatten == False
#         else np.dstack((x, y, z, r, g, b, l)).reshape(-1, 7)
#     )
#     # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
#     return xyzrgbl


def CreateBag():  # img,imu, bagname, timestamps
    """read time stamps"""
    # 所转化场景的根目录
    root_dir = "/media/tang/Elements/dataset/scannet/scene0010_00/"
    # 返回所有图片的位置
    imgs_depth = ReadImages(root_dir + "frames/depth/")
    imgs_color = ReadRGBImages(root_dir + "frames/color/")
    print("The length of images:", len(imgs_depth))
    imagetimestamps = []
    # 返回一个列表，每个元素为一个 4*4 数组
    posedata = ReadPose(root_dir + "frames/pose/")  # the url of IMU data
    print("The length of poses:", len(posedata))
    # 帧率 30
    imagetimestamps = np.linspace(
        0 + 1.0 / 30.0, (len(imgs_depth)) / 30.0, len(imgs_depth)
    )
    # 所创建 rosbag 的目录和名称
    bag = rosbag.Bag("/media/tang/Elements/dataset/scannet/scene_0010_00.bag", "w")

    """rosbag 写 tf 消息、点云消息、rgb 消息"""
    try:
        """tf 消息"""
        for i in range(len(posedata)):

            """posestamped 消息"""

            """tf 消息"""
            tf_oxts_msg = TFMessage()
            tf_oxts_transform = TransformStamped()
            oxts_tf = Transform()
            tf_oxts_transform.header.stamp = rospy.rostime.Time.from_sec(
                float(imagetimestamps[i])
            )
            tf_oxts_transform.header.frame_id = "/world"
            tf_oxts_transform.child_frame_id = "/camera"
            pose_numpy = posedata[i]
            # 坐标直接写入
            oxts_tf.translation.x = pose_numpy[0, 3]
            oxts_tf.translation.y = pose_numpy[1, 3]
            oxts_tf.translation.z = pose_numpy[2, 3]
            # 四元数由旋转矩阵得到
            rotate_numpy = pose_numpy[0:3, 0:3]
            # print(np.linalg.inv(rotate_numpy))
            # print(rotate_numpy)
            # q = Quaternion(matrix=rotate_numpy)
            q = tf.transformations.quaternion_from_matrix(pose_numpy)
            oxts_tf.rotation.x = q[0]
            oxts_tf.rotation.y = q[1]
            oxts_tf.rotation.z = q[2]
            oxts_tf.rotation.w = q[3]
            tf_oxts_transform.transform = oxts_tf
            tf_oxts_msg.transforms.append(tf_oxts_transform)
            bag.write("/tf", tf_oxts_msg, tf_oxts_transform.header.stamp)
            bag.write("/camera_pose",tf_oxts_transform,tf_oxts_transform.header.stamp)

        for i in range(len(imgs_depth)):
            print("Adding %s" % imgs_depth[i])
            # 读入 color
            raw_color = cv2.imread(imgs_color[i])
            resize_color = cv2.resize(raw_color, (640, 480))
            # cv2.imshow('1',resize_color)
            # cv2.waitKey(1000)

            """写入 resize 成 (640,480) RGB"""
            # bridge = CvBridge()
            # encoding = 'bgr8'
            # header = Header()
            # header.frame_id = "/camera"
            # header.stamp = rospy.Time.from_sec(float(imagetimestamps[i]))
            # image_message = bridge.cv2_to_imgmsg(resize_color, encoding=encoding)
            # image_message.header.frame_id = header.frame_id
            # bag.write('/camera_color', image_message, header.stamp)

            """写入点云，XYZRGB"""
            # opencv 的色彩读入顺序为 BGR
            (B, G, R) = cv2.split(resize_color)
            raw_depth = cv2.imread(imgs_depth[i], -1)
            [H, W] = raw_depth.shape
            raw_depth_np = raw_depth.astype(np.float32)
            pc = depth2xyzrgb(
                raw_depth_np,
                B,
                G,
                R,
                577.870605,
                577.870605,
                319.500000,
                239.500000,
                flatten=True,
            )
            header = Header()
            header.frame_id = "/camera"
            header.stamp = rospy.Time.from_sec(float(imagetimestamps[i]))
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("rgb", 12, PointField.UINT32, 1),
            ]
            pcl_msg = pcl2.create_cloud(header, fields, pc)
            bag.write("/camera_point", pcl_msg, pcl_msg.header.stamp)

    finally:
        bag.close()


if __name__ == "__main__":
    # 分别输入图像的文件夹、轨迹的文件夹、要保存的包位置
    # print(sys.argv)
    CreateBag()
