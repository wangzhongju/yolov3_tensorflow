#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import core.utils as utils
import core.utils_new as utils_new
import tensorflow as tf
from PIL import Image
import os
import os.path as osp
import glob
import shutil
from core.config import cfg
import time


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# save label all
def coco_bbox(image_path, save_path, return_tensorsm, graph):
    t1 = time.time()
    # return_elements     = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    input_size          = cfg.TEST.INPUT_SIZE
    # graph               = tf.Graph()
    save_file_name      = osp.basename(image_path).split('.jp')[0] + '.txt'
    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data          = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data          = image_data[np.newaxis, ...]
    # pb_file             = cfg.TEST.PB_FILE
    name_dict           = utils.read_class_names(cfg.TEST.CLASSES)
    num_classes         = len(name_dict)
    choose_classes      = cfg.TEST.CHOOSE_CLASSES

    with open(osp.join(save_path, save_file_name), 'w') as fp:
        # return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
        
        with tf.compat.v1.Session(graph=graph) as sess:
            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})
        # t2 = time.time()
        # print("time2: ", t2 - t1)
        
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
        
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.45)
        bboxes = utils.nms(bboxes, 0.35, method='nms')
        o = str(' '+str(0))
        # with open(osp.join(save_path, save_file_name), 'w') as fp:
        for bbox in bboxes:
            min_x, min_y, max_x, max_y = [str(int(bbox[i])) for i in range(4)]
            score = str(round(bbox[4], 2))
            label_name = name_dict[int(bbox[5])]
            if label_name == 'person':
                label_name_choose = 'Pedestrian'
            elif label_name == 'bicycle' or label_name == 'motorbike':
                label_name_choose = 'Cyclist'
            elif label_name == 'car' or label_name == 'bus' or label_name == 'truck':
                label_name_choose = 'Car'
            else:
                label_name_choose = 'DontCare'
            if label_name in choose_classes:
                fp.writelines(label_name+o+o+o+' '+min_x+' '+min_y+' '+max_x+' '+max_y+o+o+o+o+o+o+o+" " +score)
                fp.writelines('\n')
        # image = utils.draw_bbox(original_image, bboxes,name_dict)
        # image = Image.fromarray(image)
        # image.show()
    return original_image, bboxes, name_dict



def alone_bbox(image_path, save_path):
    class_list          = ["bench", "roadblock", "babycar", "wheelchair"]
    return_elements     = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    input_size          = cfg.TEST.INPUT_SIZE
    graph               = tf.Graph()
    save_file_name      = osp.basename(image_path).split('.')[0] + '.txt'
    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data          = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data          = image_data[np.newaxis, ...]

    with open(osp.join(save_path, save_file_name), 'a') as fp:
        name_dict_list = []
        bboxes_list = []
        for newlabel in class_list:
            pb_file     = "./yolov3_{}.pb".format(newlabel)
            name_dict   = utils.read_class_names("./data/classes/{}.names".format(newlabel))
            name_dict_list.append(name_dict)
            num_classes = len(name_dict)
            
            return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

            with tf.Session(graph=graph) as sess:
                pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                    [return_tensors[1], return_tensors[2], return_tensors[3]],
                            feed_dict={ return_tensors[0]: image_data})
            
            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                        np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
            
            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.9)
            bboxes = utils.nms(bboxes, 0.90, method='nms')
            bboxes_list.append(bboxes)
            o = str(' '+str(0))
            for bbox in bboxes:
                min_x, min_y, max_x, max_y = [str(int(bbox[i])) for i in range(4)]
                label_name = name_dict[int(bbox[5])]
                fp.writelines(label_name+o+o+o+' '+min_x+' '+min_y+' '+max_x+' '+max_y+o+o+o+o+o+o+o)
                fp.writelines('\n')
        # for i,bboxes_ in enumerate(bboxes_list):
        #     image = utils.draw_bbox(original_image, bboxes_, name_dict_list[i])
        # image = Image.fromarray(image)
        # image.show()
    return bboxes_list, name_dict_list



if __name__ == "__main__":
    pb_file         = cfg.TEST.PB_FILE
    graph           = tf.Graph()
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    # input_size      = cfg.TEST.INPUT_SIZE
    # save_file_name      = osp.basename(image_path).split('.jp')[0] + '.txt'
    # original_image      = cv2.imread(image_path)
    # original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # original_image_size = original_image.shape[:2]
    # image_data          = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    # image_data          = image_data[np.newaxis, ...]
    # name_dict           = utils.read_class_names(cfg.TEST.CLASSES)
    # num_classes         = len(name_dict)
    # choose_classes      = cfg.TEST.CHOOSE_CLASSES

    realPath        = os.getcwd()
    labelPath       = osp.join(realPath, 'label2d')
    imgPath         = osp.join(realPath, 'imgorg')
    newimg          = osp.join(realPath, 'img2d')
    files           = os.listdir(imgPath)
    files.sort()

    for file in files:
        images_path = osp.join(imgPath, file)
        images = glob.glob(images_path+'/*')
        save_path = osp.join(labelPath, file)
        save_img = osp.join(newimg, file)
        if osp.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        if osp.exists(save_img):
            shutil.rmtree(save_img)
        os.makedirs(save_img)
        # if not osp.exists(save_path) or not osp.exists(save_img):
        for image_path in images:
            start     = time.time()
            img_name  = osp.basename(image_path)
            original_image, bboxes, name_dict = coco_bbox(image_path, save_path, return_tensors, graph)
            # original_image, bboxes, name_dict = coco_bbox(image_path, save_path)
            end       = time.time()
            print("detect a image usage time = ", end - start)
            # bboxes_list, name_dict_list = alone_bbox(image_path, save_path)
            # bboxes_list.append(bboxes)
            # name_dict_list.append(name_dict)
            # for i,bboxes_ in enumerate(bboxes_list):
            #     image = utils_new.draw_bbox(original_image, bboxes_, name_dict_list[i])

            image = utils_new.draw_bbox(original_image, bboxes, name_dict)
            image_ = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(osp.join(save_img, img_name), image_)
            # image = Image.fromarray(image)
            # image.show()


