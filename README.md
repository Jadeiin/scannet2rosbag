# scannet2rosbag
A python2.7 inplementation for converting the Scannet dataset into a rosbag containing depth point clouds(with colors representing the corresponding semantic labels ) and RGB images.

## Usage
You need to replace these lines with your own directory:

1.  [Line 161](https://github.com/BIT-TYJ/scannet2rosbag/blob/ba38694a8e53d99441d0534d701f992d48d81ce5/to_bag-xyzrgbl-600.py#L161)

```
root_dir = "/media/tang/Elements/dataset/scannet/scene0010_00/"
```
2.  [Line 174](https://github.com/BIT-TYJ/scannet2rosbag/blob/ba38694a8e53d99441d0534d701f992d48d81ce5/to_bag-xyzrgbl-600.py#L174)
```
bag = rosbag.Bag("/media/tang/Elements/dataset/scannet/scene_0010_00.bag", 'w')
```

