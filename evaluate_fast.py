from __future__ import division, print_function, absolute_import

import os
import sys
import numpy as np
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.layers import Input
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from PIL import Image
from keras.models import Model

cuhk = {}
cuhk['TEST'] = os.path.join('../dataset/CUHK03', 'bbox_test')
cuhk['TEST_NUM'] = 5332
cuhk['QUERY'] = os.path.join('../dataset/CUHK03', 'query')
cuhk['QUERY_NUM'] = 1400

market = {}
market['TEST'] = os.path.join('../dataset/Market', 'bounding_box_test')
market['TEST_NUM'] = 19732
market['QUERY'] = os.path.join('../dataset/Market', 'query')
market['QUERY_NUM'] = 3368

duke = {}
duke['TEST'] = os.path.join('../dataset/Duke', 'bounding_box_test')
duke['TEST_NUM'] = 17661
duke['QUERY'] = os.path.join('../dataset/Duke', 'query')
duke['QUERY_NUM'] = 2228


def extract_feature(dir_path, net):
  features = []
  infos = []
  num = 0
  for image_name in os.listdir(dir_path):
    arr = image_name.split('_')
    person = int(arr[0])
    camera = int(arr[1][1])
    image_path = os.path.join(dir_path, image_name) 
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = net.predict(x)
    features.append(np.squeeze(feature))
    infos.append((person, camera))

  return features, infos

# use GPU to calculate the similarity matrix
query_t = tf.placeholder(tf.float32, (None, None))
test_t = tf.placeholder(tf.float32, (None, None))
query_t_norm = tf.nn.l2_normalize(query_t, dim=1)
test_t_norm = tf.nn.l2_normalize(test_t, dim=1)
tensor = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

def map_rank_quick_eval(query_info, test_info, result_argsort):
    # about 10% lower than matlab result
    # for evaluate rank1 and map
    match = []
    junk = []

    for q_index, (qp, qc) in enumerate(query_info):
        tmp_match = []
        tmp_junk = []
        for t_index in range(len(test_info)):
            p_t_idx = result_argsort[q_index][t_index]
            p_info = test_info[int(p_t_idx)]

            tp = p_info[0]
            tc = p_info[1]
            if tp == qp and qc != tc:
                tmp_match.append(t_index)
            elif tp == qp or tp == -1:
                tmp_junk.append(t_index)
        match.append(tmp_match)
        junk.append(tmp_junk)

    rank_1 = 0.0
    mAP = 0.0
    for idx in range(len(query_info)):
        if idx % 100 == 0:
            print('evaluate img %d' % idx)
        recall = 0.0
        precision = 1.0
        ap = 0.0
        YES = match[idx]
        IGNORE = junk[idx]
        ig_cnt = 0
        for ig in IGNORE:
            if ig < YES[0]:
                ig_cnt += 1
            else:
                break
        if ig_cnt >= YES[0]:
            rank_1 += 1

        for i, k in enumerate(YES):
            ig_cnt = 0
            for ig in IGNORE:
                if ig < k:
                    ig_cnt += 1
                else:
                    break
            cnt = k + 1 - ig_cnt
            hit = i + 1
            tmp_recall = hit / len(YES)
            tmp_precision = hit / cnt
            ap = ap + (tmp_recall - recall) * ((precision + tmp_precision) / 2)
            recall = tmp_recall
            precision = tmp_precision

        mAP += ap
    rank1_acc = rank_1 / QUERY_NUM
    mAP = mAP / QUERY_NUM
    print('Rank 1:\t%f' % rank1_acc)
    print('mAP:\t%f' % mAP)
    return rank1_acc, mAP

def evaluate(dataset, model):
  TEST = dataset['TEST']
  TEST_NUM = dataset['TEST_NUM']
  QUERY = dataset['QUERY']
  QUERY_NUM = dataset['QUERY_NUM']
  # load model
  net = load_model(model)
  net = Model(input=net.input, output=net.get_layer('avg_pool').output)

  test_f, test_info = extract_feature(TEST, net)
  query_f, query_info = extract_feature(QUERY, net)
  
  result = sess.run(tensor, {query_t: query_f, test_t: test_f})
  result_argsort = np.argsort(result, axis=1)
  
  rank1, mAP = map_rank_quick_eval(query_info, test_info, result_argsort)

  print ('%s - %s'%(model, TEST)) 
  print ('1: %f\tmAP: %f'%(rank_1, mAP))
  print ()
  sys.stdout.flush()

for i in range(1, 26):
  model_a = 'checkpoint_a/%d.ckpt'%i
  model_b = 'checkpoint_b/%d.ckpt'%i
  dataset = duke
  evaluate(dataset, model_a)
  evaluate(dataset, model_b)

