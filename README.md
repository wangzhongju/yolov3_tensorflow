## part 1. Quick start
1. **Download this file**
````bashrc
$ git clone 'https://github.com/wangzhongju/yolov3_tensorflow.git'
````
2. **place your data folder in '{project}/imgorg/'**

>e.g.  there is a folder named 'test' what contained some images,
now, wo can do follow this steps:
````bashrc
$ mv /path/to/test/folder {project}/imgorg
$ python cascade_detection.py
````
>At last, you can see the results in '{project}/label2d'

## part 2. Retrain
