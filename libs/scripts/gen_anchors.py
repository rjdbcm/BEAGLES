"""
Created on Feb 20, 2017
@author: jumabek
"""
import os
import argparse
import numpy as np
import sys
import os
import random
np.seterr(invalid='raise')
width_in_cfg_file = 416.
height_in_cfg_file = 416.


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


def write_anchors_to_file(centroids, X, anchor_file):
    with open(anchor_file, 'w') as f:

        anchors = centroids.copy()
        print(anchors.shape)

        for i in range(anchors.shape[0]):
            anchors[i][0] *= width_in_cfg_file / 32.
            anchors[i][1] *= height_in_cfg_file / 32.

        widths = anchors[:, 0]
        sorted_indices = np.argsort(widths)

        print('Anchors =\n', anchors[sorted_indices])

        for i in sorted_indices[:-1]:
            f.write('%0.2f,%0.2f, ' % (anchors[i, 0], anchors[i, 1]))

        # there should not be comma after last anchor, that's why
        f.write('%0.2f,%0.2f\n' % (anchors[sorted_indices[-1:], 0],
                                   anchors[sorted_indices[-1:], 1]))

        # f.write('%f\n' % (avg_IOU(X, centroids)))


def kmeans(X, centroids, anchor_file):
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

        print("iter {}: dists = {}".format(iter, dists))

        # assign samples to centroids
        assignments = np.argmin(D, axis=1)

        if (assignments == prev_assignments).all():
            print("Centroids =\n", centroids)
            write_anchors_to_file(centroids, X, anchor_file)
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

        prev_assignments = assignments.copy()
        old_D = D.copy()

        if np.isnan(dists):
            print("error", file=sys.stderr)
            exit(1)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='yolo/filelist.txt',
                        help='path to filelist\n')
    parser.add_argument('--output', default='anchors', type=str,
                        help='Output anchor directory\n')
    parser.add_argument('--num_clusters', default=5, type=int,
                        help='number of clusters\n')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    with open(args.filelist) as f:
        lines = [line.rstrip('\n') for line in f.readlines()]

    annotation_dims = []

    for line in lines:
        # this is awful why?!?!
        # line = line.replace('images','labels')
        # line = line.replace('img1','labels')
        # line = line.replace('JPEGImages', 'labels')

        line = line.replace('.jpg' or '.png', '.txt')
        # line = line.replace('.png', '.txt')
        print(line)
        with open(line) as f2:
            for line in f2.readlines():
                line = line.rstrip('\n')
                w, h = line.split(' ')[3:]
                # print(w,h)
                annotation_dims.append(tuple(map(float, (w, h))))
    annotation_dims = np.array(annotation_dims)

    if args.num_clusters == 0:
        for num_clusters in range(1, 11):  # we make 1 through 10 clusters
            anchor_file = os.path.join(args.output, 'anchors%d.txt' % num_clusters)

            indices = [random.randrange(annotation_dims.shape[0]) for i in
                       range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims, centroids, anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = os.path.join(args.output, 'anchors%d.txt' % args.num_clusters)
        indices = [random.randrange(annotation_dims.shape[0]) for i in
                   range(args.num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims, centroids, anchor_file)
        print('centroids.shape', centroids.shape)


if __name__ == "__main__":
    main(sys.argv)

