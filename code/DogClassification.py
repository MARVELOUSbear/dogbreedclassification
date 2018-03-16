# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:13:23 2017

@author: ALW15
"""

import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications import xception
from keras.applications import inception_v3
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

NUM_CLASSES = 16
SEED = 1987
        
def read_img(img_id, train_or_test, size,data_dir):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img

class DogClassification:
    model=LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
    breed_list=[]
    
    def train_network(self):
        INPUT_SIZE = 224
        data_dir = '../input'
        labels = pd.read_csv(join(data_dir, 'labels.csv'))
        selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)
        self.breed_list=selected_breed_list
        labels = labels[labels['breed'].isin(selected_breed_list)]
        labels['target'] = 1
        labels['rank'] = labels.groupby('breed').rank()['id']
        labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
        np.random.seed(seed=SEED)
        rnd = np.random.random(len(labels))
        train_idx = rnd < 0.8
        valid_idx = rnd >= 0.8
        y_train = labels_pivot[selected_breed_list].values
        ytr = y_train[train_idx]
        yv = y_train[valid_idx]
        INPUT_SIZE = 299
        POOLING = 'avg'
        x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
        for i, img_id in tqdm(enumerate(labels['id'])):
            img = read_img(img_id, 'train', (INPUT_SIZE, INPUT_SIZE),data_dir)
            x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
            x_train[i] = x
        Xtr = x_train[train_idx]
        Xv = x_train[valid_idx]
        xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
        train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
        valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)
        Xtr = x_train[train_idx]
        Xv = x_train[valid_idx]
        inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=POOLING)
        train_i_bf = inception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
        valid_i_bf = inception_bottleneck.predict(Xv, batch_size=32, verbose=1)
        X = np.hstack([train_x_bf, train_i_bf])
        V = np.hstack([valid_x_bf, valid_i_bf])
        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
        logreg.fit(X, (ytr * range(NUM_CLASSES)).sum(axis=1))
        self.model=logreg
        valid_probs = logreg.predict_proba(V)
        valid_preds = logreg.predict(V)
        logloss=log_loss(yv, valid_probs)
        accuracy=accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)
        joblib.dump(self.model,'./Lr.model')
        list_str=str(self.breed_list)
        list_str=list_str.replace("[","")
        list_str=list_str.replace("]","")
        file=open('./code/breed_list.data','w')
        file.write(list_str)
        file.close()
        return logloss,accuracy

    def predictNewPic(self,img_dir):
        INPUT_SIZE = 299
        POOLING = 'avg'
        file = open("./code/breed_list.data")
        self.breed_list = file.read().split(",")
        x_train = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
        img = image.load_img(img_dir, target_size=(INPUT_SIZE, INPUT_SIZE))
        img = image.img_to_array(img)
        x_train = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
        Xv = x_train
        xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
        valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)
        inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=POOLING)
        valid_i_bf = inception_bottleneck.predict(Xv, batch_size=32, verbose=1)
        V = np.hstack([valid_x_bf, valid_i_bf])
        logreg=joblib.load('./code/Lr.model')
        valid_preds = logreg.predict(V)
        class_name=self.breed_list[int(valid_preds[0])]
        return class_name

