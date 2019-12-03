## part 1. Quick start
1. **Download this file**
````bashrc
$ git clone 'https://github.com/wangzhongju/yolov3_tensorflow.git'
````
2. **place your data folder in '{project}/imgorg/'**

> e.g.  there is a folder named 'test' what contained some images,
now, wo can do follow this steps:
````bashrc
$ mv /path/to/test/folder {project}/imgorg
$ python cascade_detection.py
````
> At last, you can see the results in '{project}/label2d'

## part 2. Retrain
1. format of txt_file:
> img1_path bbox_1,labelname bbox_2,labelname ...
img_path: the address of image, 
***e.g. /path/for/test/test.jpg***
bbox: x1,y1,x2,y2
labelname: locate in ***'{project}/data/classes/train.names'***, 
you can rewrite it.
therefore, every line of the txt_file just like:
**/path/to/test.jpg 23,145,342,32,1 535,232,23,56,0**
there are two bboxes in picture named test.jpg, (23,145,342,32) and (535,232,23,56), one bbox labelname 1, the other labelname 0

2. check './core/config.py', make sure the path correct.
3. run command
````bashrc
$ python train.py
````
4. **convert '.data-00000-of-00001' to '.pb'**
At the end of trianing, wo got some files located in '{project}/checkpoint/', e.g. one of them named 'yolov3_test_loss=12.342.ckpt-10.data-00000-of-00001', next, open the file **'{project}/freeze_gragh.py'**, confirm that variable **'ckpt_file'** have the same name with 'yolov3_test_loss=12.342.ckpt-10.data-00000-of-00001', run:
````bashrc
$ python freeze_graph.py
````
result: see the variable **'pb_file'** 