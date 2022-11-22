# scannet2rosbag
A python2.7 inplementation for converting the Scannet dataset into a rosbag, which contains depth point clouds(with colors representing the corresponding semantic labels ) and RGB images.

## Usage
You need to replace these lines with your own directory:

1.  [Line 154](https://github.com/BIT-TYJ/scannet2rosbag/blob/f4fe5b264ff6a0ccef0141fb4d27db6232964008/to_bag-xyzl.py#L154)

```
root_dir = "/media/tang/Elements/dataset/scannet/scene0010_00/"
```
2.  [Line 167](https://github.com/BIT-TYJ/scannet2rosbag/blob/f4fe5b264ff6a0ccef0141fb4d27db6232964008/to_bag-xyzl.py#L167)
```
bag = rosbag.Bag("/media/tang/Elements/dataset/scannet/scene_0010_00.bag", 'w')
```

