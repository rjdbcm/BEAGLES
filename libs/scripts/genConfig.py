import configparser
import numpy as np
import sys
import os
import random
from collections import OrderedDict
import xml.etree.ElementTree as ET
import re

np.seterr(invalid='raise')


class Sections(OrderedDict):
    """
    Mangles section names with a number for editing duplicate sections with
    config parser. Non-duplicate entries retain their original name.
    """
    _unique = 0

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            if key not in ["net", "region", "detection"]:
                key += str(self._unique)
            self._unique += 1
        OrderedDict.__setitem__(self, key, value)


def fix(target):
    with open(target, 'r') as f:
        content = f.read()
        replace = re.sub("(\d+)\]", r']', content)
    with open(target, 'w') as f:
        f.write(replace)


def IOU(x, centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def avg_IOU(X, centroids):
    n, d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        # note IOU() will return array which contains IoU for each centroid
        # and X[i] // slightly ineffective, but I am too lazy
        sum += max(IOU(X[i], centroids))
    return sum / n


def write_anchors_to_file(centroids, target_cfg):
    config = configparser.ConfigParser(strict=False, dict_type=Sections)
    config.read(target_cfg)

    width_in_cfg_file = float(config.get('net', 'width'))
    height_in_cfg_file = float(config.get('net', 'height'))

    anchors = centroids.copy()
    # print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0] *= width_in_cfg_file / 32.
        anchors[i][1] *= height_in_cfg_file / 32.

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    # print('Anchors =\n', anchors[sorted_indices])

    s = ""
    for i in sorted_indices[:-1]:
        s += ('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

    # there should not be comma after last anchor, that's why
    s += ('%0.2f,%0.2f' % (anchors[sorted_indices[-1:], 0],
                               anchors[sorted_indices[-1:], 1]))

    # write config
    config.set(config.sections()[-1], 'anchors', s)
    config.sections()
    with open(target_cfg, 'w') as f:
         config.write(f)

    # strip off number tags from duplicate sections
    fix(target_cfg)


def kmeans(X, centroids, target_cfg):
    N = X.shape[0]
    k, dim = centroids.shape
    prev_assignments = np.ones(N) * (-1)
    iter = 0
    old_D = np.zeros((N, k))

    while True:
        D = []
        iter += 1
        for i in range(N):
            d = 1 - IOU(X[i], centroids)
            D.append(d)
        D = np.array(D)  # D.shape = (N,k)

        dists = np.sum(np.abs(old_D - D))

        # print("iter {}: dists = {}".format(iter, dists))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            # print("Centroids =\n", centroids)
            write_anchors_to_file(centroids, target_cfg)
            return

        # calculate new centroids
        centroid_sums = np.zeros((k, dim), np.float)
        for i in range(N):
            centroid_sums[assignments[i]] += X[i]
        for j in range(k):
            try:
                centroids[j] = centroid_sums[j] / (np.sum(assignments == j))
            except FloatingPointError:
                pass

        prev_assignments = assignments[:]
        old_D = D[:]

        if np.isnan(dists):
            print("error", file=sys.stderr)
            exit(1)


def convertBbox(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    w = box[1] - box[0]
    h = box[3] - box[2]
    w = w*dw
    h = h*dh
    return w, h


def setNumClassesYOLOv2(target_cfg, num_classes):
    config = configparser.ConfigParser(strict=False, dict_type=Sections)
    config.read(target_cfg)
    go = True
    # get the section just before region
    while go:
        for section in config.sections():
            if section.startswith('region'):
                go = False
                break
            _section = section
    config.set('region', 'classes', str(num_classes))
    config.set(_section, 'filters', str(5 * (num_classes + 5)))
    with open(target_cfg, 'w') as f:
        config.write(f)
    fix(target_cfg)


def genConfigYOLOv2(folder, target_cfg, num_clusters, num_classes):
    target_cfg = os.path.abspath(target_cfg)
    annotation_dims = []

    for file in os.listdir(folder):
        if file.endswith(".xml"):
            tree = ET.parse(folder + file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                # cls = obj.find('name').text
                # if cls not in classes or int(difficult) == 1:
                #     continue
                # cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text),
                     float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))
                bb = convertBbox((w, h), b)
                w, h = bb
                annotation_dims.append(tuple(map(float, (w, h))))
    annotation_dims = np.array(annotation_dims)

    if num_clusters == 0:
        for clusters in range(1, 11):  # we make 1 through 10 clusters
            indices = [random.randrange(annotation_dims.shape[0]) for i in
                       range(clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims, centroids, target_cfg)
            # print('centroids.shape', centroids.shape)
    else:
        indices = [random.randrange(annotation_dims.shape[0]) for i in
                   range(num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims, centroids, target_cfg)
        # print('centroids.shape', centroids.shape)

    setNumClassesYOLOv2(target_cfg, num_classes)


genConfigYOLOv2("../../data/committedframes/", "../../data/cfg/tiny-yolov2.cfg", 5, 4)
