# -*- coding:UTF-8 -*-
import cv2
import random
import time, sys, os
from ros import rosbag
import roslib
import rospy
roslib.load_manifest('sensor_msgs')
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
import pandas as pd
from scipy import misc
from PIL import ImageFile
from PIL import Image as ImagePIL
import matplotlib.pyplot as pyplot
import collections

def random2RGB(rgb_num):   
    r=[]
    g=[]
    b=[]
    for i in range(rgb_num+1):               #0~rgb_num
        if i == 0:   #未知语义用0,0,0的黑色表示
            r.append(0)
            g.append(0)
            b.append(0)
            continue
        r.append(np.random.randint(256))
        g.append(np.random.randint(256))
        b.append(np.random.randint(256))
    return r,g,b

'''读语义文件'''
tsv_sem = pd.read_csv('scannetv2-labels.combined.tsv', sep='\t')
sem_value_1000 = tsv_sem['id']
sem_value_1000_index = [i for i in range(len(sem_value_1000))]
# sem_value_nyu40 = tsv_sem['nyu40id']
sem_meaning = tsv_sem['category']
sem_value_1000_dict = dict(zip(sem_value_1000, sem_value_1000_index))
# #投影：1000多种的语义label 投影成40种类别的语义label :dict_sem_value={0:0, ... , 1357:40}
# dict_sem_value = dict(zip(sem_value_1000, sem_value_nyu40))   #print(dict_sem_value[1357])
# dict_sem_value_modify = collections.defaultdict(list)
# for k, v in dict_sem_value.iteritems():
#     dict_sem_value_modify[k].append(v)

#600多种sem_label对应的随机RGB值
r_sem,g_sem,b_sem = random2RGB(len(sem_value_1000))      # RGB_sem = [[0,200,0],...,[10,20,10]] 

def ReadImages(filename):
    all = []
    files = os.listdir(filename)
    all_num = 0
    for f in sorted(files):
        if os.path.splitext(f)[1] in ['.png']:
            all_num = all_num+1
    for i in range(all_num):
        all.append(filename+str(i)+'.png')
    return all

def ReadRGBImages(filename):
    all = []
    files = os.listdir(filename)
    all_num = 0
    for f in sorted(files):
        if os.path.splitext(f)[1] in ['.jpg']:
            all_num = all_num+1
    for i in range(all_num):
        all.append(filename+str(i)+'.jpg')
    return all

def ReadPose(filename):
    file = open(filename,'r')
    all = file.readlines()
    posedata = []
    for f in all:
        line = f.rstrip('\n').split(' ')
        line = np.array(line).astype(float).reshape(4,4)
        posedata.append(line)
    return posedata

def depth2xyz(depth_map,fx,fy,cx,cy,flatten=False,depth_scale=1000):
    h,w=np.mgrid[0:depth_map.shape[0],0:depth_map.shape[1]]
    z=depth_map/depth_scale
    x=(w-cx)*z/fx
    y=(h-cy)*z/fy
    xyz=np.dstack((x,y,z)) if flatten==False else np.dstack((x,y,z)).reshape(-1,3)
    #xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz

def depth2xyzl(depth_map,semantic_image,fx,fy,cx,cy,flatten=False,depth_scale=1000):
    h,w=np.mgrid[0:depth_map.shape[0],0:depth_map.shape[1]]
    z=depth_map/depth_scale
    x=(w-cx)*z/fx
    y=(h-cy)*z/fy

    r = np.ones((480,640))
    g = np.ones((480,640))
    b = np.ones((480,640))
    # sem_touying = np.ones((480,640))
    for i in range(0,480):
        for j in range(0,640):
            #原始的语义标签semantic_image[i][j]: 0~1357
            sem_label = semantic_image[i][j]
            #投影的语义标签a[i][j]: 0~40
            if sem_value_1000_dict.has_key(sem_label):
                sem_index = sem_value_1000_dict[sem_label]
            else:
                sem_index = 0
            # if sem_label  not in list(sem_value_1000):
            #     sem_index_in_sem_label_list = 0
            # else:
            #     sem_index_in_sem_label_list = list(sem_value_1000).index(sem_label) # 0表示未知语义的物体
            r[i][j] = r_sem[sem_index]
            g[i][j] = g_sem[sem_index]
            b[i][j] = b_sem[sem_index]

    # print(sem_touying[1][320],"第一行最中间")
    # print(sem_touying[240][250],"第一行最中间")
    new_im = ImagePIL.fromarray(b)     #调用Image库，数组归一化 
    misc.imsave('new_img.jpg', new_im)
    xyzrgbl=np.dstack((x,y,z,r,g,b)) if flatten==False else np.dstack((x,y,z,r,g,b)).reshape(-1,6)
    return xyzrgbl

def depth2xyzrgbl(depth_map,B,G,R,semantic_image,fx,fy,cx,cy,flatten=False,depth_scale=1000):
    h,w=np.mgrid[0:depth_map.shape[0],0:depth_map.shape[1]]
    z=depth_map/depth_scale
    x=(w-cx)*z/fx
    y=(h-cy)*z/fy
    # r = R
    # b = B
    # g =  G
    # print(r[100,:])
    a = np.ones((480,640))
    for i in range(0,480):
        for j in range(0,640):
            a[i][j] = 255
    r = a - R
    b = a - B
    g = a - G

    l = semantic_image
    xyzrgbl=np.dstack((x,y,z,r,g,b,l)) if flatten==False else np.dstack((x,y,z,r,g,b,l)).reshape(-1,7)
    #xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyzrgbl



def CreateBag():#img,imu, bagname, timestamps
    
    
    '''read time stamps'''
    # 所转化场景的根目录
    root_dir = "/media/tang/Elements/dataset/scannet/scene0010_00/"
    # 返回所有图片的位置
    imgs_depth = ReadImages(root_dir+"frames/depth/")
    imgs_color = ReadRGBImages(root_dir+"frames/color/")
    imgs_semantic = ReadImages(root_dir+"frames/semantic/")
    print("The length of images:",len(imgs_depth))
    imagetimestamps=[]
    # 返回一个列表，每个元素为一个4*4数组
    posedata = ReadPose(root_dir+"traj.txt") #the url of  IMU data
    print("The length of poses:",len(posedata))
    # 帧率30
    imagetimestamps = np.linspace(0+1.0/30.0,(len(imgs_depth))/30.0, len(imgs_depth))
    # 所创建rosbag的目录和名称
    bag = rosbag.Bag("/media/tang/Elements/dataset/scannet/scene_0010_00.bag", 'w')

    '''rosbag写tf消息、点云消息、rgb消息'''
    try:    
        '''tf消息'''
        for i in range(len(posedata)):

            '''posestamped消息'''

            '''tf消息'''
            tf_oxts_msg = TFMessage()
            tf_oxts_transform = TransformStamped() 
            oxts_tf = Transform()           
            tf_oxts_transform.header.stamp = rospy.rostime.Time.from_sec(float(imagetimestamps[i]))
            tf_oxts_transform.header.frame_id = '/world'
            tf_oxts_transform.child_frame_id = '/camera'
            pose_numpy = posedata[i]
            # 坐标直接写入
            oxts_tf.translation.x = pose_numpy[0,3]
            oxts_tf.translation.y = pose_numpy[1,3]
            oxts_tf.translation.z = pose_numpy[2,3]
            # 四元数由旋转矩阵得到
            rotate_numpy = pose_numpy[0:3,0:3]
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
            # bag.write("/camera_pose",tf_oxts_transform,tf_oxts_transform.header.stamp)
        
        for i in range(len(imgs_depth)):
            print("Adding %s" % imgs_depth[i])
            #读入semantic
            raw_semantic = cv2.imread(imgs_semantic[i], -1)
            resize_semantic = cv2.resize(raw_semantic,(640,480))
            #读入color
            raw_color = cv2.imread(imgs_color[i])
            resize_color = cv2.resize(raw_color,(640,480))
            # cv2.imshow('1',resize_color)
            # cv2.waitKey(1000)

            '''写入resize成 (640,480) RGB'''
            bridge = CvBridge()
            encoding = 'bgr8'
            header = Header()
            header.frame_id = "/camera"
            header.stamp = rospy.Time.from_sec(float(imagetimestamps[i]))
            image_message = bridge.cv2_to_imgmsg(resize_color, encoding=encoding)
            image_message.header.frame_id = header.frame_id
            bag.write('/camera_color', image_message, header.stamp)

            '''写入点云, XYZRGBL'''
            #opencv的色彩读入顺序为BGR
            (B, G, R) = cv2.split(resize_color)
            raw_depth = cv2.imread(imgs_depth[i], -1)
            [H, W] = raw_depth.shape
            raw_depth_np = raw_depth.astype(np.float32)
            pc = depth2xyzl(raw_depth_np,resize_semantic, 577.870605, 577.870605, 319.500000, 239.500000, flatten=True)
            header = Header()
            header.frame_id = "/camera"
            header.stamp = rospy.Time.from_sec(float(imagetimestamps[i]))
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('r', 12, PointField.FLOAT32, 1),
                    PointField('g', 16, PointField.FLOAT32, 1),
                    PointField('b', 20, PointField.FLOAT32, 1)]
            pcl_msg = pcl2.create_cloud(header, fields, pc)
            bag.write('/camera_point', pcl_msg, pcl_msg.header.stamp)


    finally:
        bag.close()

if __name__ == "__main__":
    # 分别输入图像的文件夹、轨迹的文件夹、要保存的包位置
    # print(sys.argv)
    CreateBag()