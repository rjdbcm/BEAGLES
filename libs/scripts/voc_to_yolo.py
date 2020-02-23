'''
This script converts PascalVOC xml to yolo txt files
'''

import glob
import argparse
import sys
import os
import xml.etree.ElementTree as ET


def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)

    return image_list


def convertBbox(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x, y, w, h


def convertAnnotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]
    list_file = open(dir_path + '.txt', 'w')

    try:
        in_file = open(dir_path + '/' + basename_no_ext + '.xml')
        list_file.write(image_path + '\n')
        list_file.close()
    except FileNotFoundError:
        print("No annotations found for {}. Skipping.".format(basename))
        return
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convertBbox((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join(
            [str(a) for a in bb]) + '\n')


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classfile', type=str, required=True)
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    global classes
    with open(args.classfile, 'r') as file:
        classes = [line.rstrip() for line in file]
    dir = [args.dir]

    for dir_path in dir:
        full_dir_path = os.path.abspath(dir_path)
        print("Reading annotations from: ", full_dir_path)
        output_path = os.path.join(full_dir_path, 'yolo/')
        print("Outputting annotations to: ", output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        image_paths = getImagesInDir(full_dir_path)

        for image_path in image_paths:
            convertAnnotation(full_dir_path, output_path, image_path)

        print("Finished processing: " + dir_path)


if __name__ == "__main__":
    main(sys.argv)
