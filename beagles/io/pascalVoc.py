#!/usr/bin/env python
# -*- coding: utf8 -*-
from defusedxml import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree  # nosec
import codecs
from beagles.base.constants import DEFAULT_ENCODING, XML_EXT


class PascalVocWriter:
    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown'):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.verified = False
        self.boxes = list()

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        pretty_root = etree.tostring(root, pretty_print=True, encoding=DEFAULT_ENCODING)
        return pretty_root.replace("  ".encode(), "\t".encode())

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def appendObjects(self, top):
        for each_object in self.boxes:
            object_item = SubElement(top, 'object')
            truncated = SubElement(object_item, 'truncated')
            height = int(float(self.imgSize[0]))
            width = int(float(self.imgSize[1]))
            minx = int(float(each_object.xmin))
            miny = int(float(each_object.ymin))
            maxy = int(float(each_object.ymax))
            maxx = int(float(each_object.xmax))
            truncated.text = "1" if maxy == height or miny == 1 else "0"
            truncated.text = "1" if maxx == width or minx == 1 else "0"
            name = SubElement(object_item, 'name')
            name.text = str(each_object.label)
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str(bool(each_object.difficult) & 1)
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object.xmin)
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object.ymin)
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object.xmax)
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object.ymax)

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=DEFAULT_ENCODING)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=DEFAULT_ENCODING)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class PascalVocReader:

    def __init__(self, filepath):
        # shapes type:
        # [label, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseXML()
        except:
            print("bewp")
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox, difficult):
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, difficult))

    def parseXML(self):
        assert self.filepath.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=DEFAULT_ENCODING)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            # Add chris
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.addShape(label, bndbox, difficult)
        return True
