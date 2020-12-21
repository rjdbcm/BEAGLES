import sys
import os
import defusedxml.ElementTree as ET
import glob
from beagles.io.flags import SharedFlagIO


def pascal_voc_clean_xml(self, annotation_dir, pick, exclusive=False):
    self.logger.info(f'Parsing {os.path.join(annotation_dir,"*.xml")} for {pick} {"exclusively" * int(exclusive)}')
    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(annotation_dir)
    annotations = glob.glob('*.xml')
    size = len(annotations)

    for i, file in enumerate(annotations):
        with open(file) as in_file:
            tree = ET.parse(in_file)
            root = tree.getroot()
            jpg = str(root.find('filename').text)
            imsize = root.find('size')
            w = int(imsize.find('width').text)
            h = int(imsize.find('height').text)
            all = list()

            for obj in root.iter('object'):
                # noinspection PyUnusedLocal
                current = list()
                name = obj.find('name').text
                if name not in pick:
                    continue

                xmlbox = obj.find('bndbox')
                xn = int(float(xmlbox.find('xmin').text))
                xx = int(float(xmlbox.find('xmax').text))
                yn = int(float(xmlbox.find('ymin').text))
                yx = int(float(xmlbox.find('ymax').text))
                current = [name, xn, yn, xx, yx]
                all += [current]

            add = [[jpg, [w, h, all]]]
            dumps += add

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]] += 1
                else:
                    stat[current[0]] = 1
    count = 0
    for i in stat:
        self.logger.info(f'\t{i}: {stat[i]}')
        count += stat[i]
    weights = {i: stat[i]/count for i in stat}
    try:
        assert count >= len(dumps), f"There are {len(dumps)} images but only {count} annotations"
    except AssertionError as e:
        self.flags.error = str(e)
        self.logger.error(str(e))
        self.send_flags()
        raise
    self.logger.info(f'\tDataset size: {len(dumps)}')

    os.chdir(cur_dir)
    return dumps, weights
