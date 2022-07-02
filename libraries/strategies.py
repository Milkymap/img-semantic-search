import cv2 

import json 
import pickle

import numpy as np 
import operator as op 
import itertools as it, functools as ft 

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

from os import path 
from glob import glob
from time import time 

from collections import Counter
from sentence_transformers import SentenceTransformer as STrans
from torchvision.utils import make_grid

from rich.progress import track 
from libraries.log import logger 

map_serializers = {
    json: ('r', 'w'),
    pickle: ('rb', 'wb')
}

env_vars = []

def measure(func):
    @ft.wraps(func)
    def _measure(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            duration = end_ if end_ > 0 else 0
            logger.debug(f"{func.__name__:<20} total execution time: {duration:04d} ms")
    return _measure

def read_image(path2image):
    cv_image = cv2.imread(path2image, cv2.IMREAD_COLOR)
    cv_image = cv2.resize(cv_image, (256, 256))
    return cv_image 

def cv2th(cv_image):
    blue, green, red = cv2.split(cv_image)
    return th.as_tensor(np.stack([red, green, blue]))

def th2cv(th_image):
    red, green, blue = th_image.numpy() # unpack
    return cv2.merge((blue, green, red))

def merge_images(path2images):
    acc = []
    for img_path in path2images:
        cv_image = read_image(img_path)
        th_image = cv2th(cv_image)
        acc.append(th_image)
    
    th_image = make_grid(acc, 4)
    return th2cv(th_image)


def pull_files(path2directory, extension='*'):
    all_paths = sorted(glob(path.join(path2directory, extension)))
    return all_paths 

def serialize(data, location, serializer):
    modes = map_serializers.get(serializer, None)
    if modes is None:
        raise Exception('serializer has to be [pickle or json]')
    with open(location, mode=modes[1]) as fp:
        serializer.dump(data, fp)
        logger.success(f'data was dumped at {location}')
    
def deserialize(location, serializer):
    modes = map_serializers.get(serializer, None)
    if modes is None:
        raise Exception('serializer has to be [pickle or json]')
    with open(location, mode=modes[0]) as fp:
        data = serializer.load(fp)
        logger.success(f'data was loaded from {location}')
    return data 

def vectorize(pil_image, vectorizer, device='cpu'):
    finger_print = vectorizer.encode(pil_image, device=device)
    return finger_print

@measure
def scoring(fingerprint, fingerprint_matrix):
    scores = fingerprint @ fingerprint_matrix.T 
    X = np.linalg.norm(fingerprint)
    Y = np.linalg.norm(fingerprint_matrix, axis=1)
    W = X * Y 
    weighted_scores = scores / W 
    return weighted_scores

def top_k(weighted_scores, k=16):
    scores = th.as_tensor(weighted_scores).float()
    _, indices = th.topk(scores, k, largest=True)
    return indices.tolist()

def load_vectorizer(path2vectotizer):
    if path.isfile(path2vectotizer):
        logger.debug('vectorizer was found | it will be load from checkpoints')
        vectorizer = deserialize(path2vectotizer, pickle)
    else:
        try:
            _, file_name = path.split(path2vectotizer)
            vectorizer = STrans(file_name)
            logger.debug('vectorizer was downloaded')
            serialize(vectorizer, path2vectotizer, pickle)
        except Exception as e:
            logger.error(e)
            raise Exception(f'can not download {file_name}')    
    return vectorizer




    

        







