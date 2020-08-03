import os
import sys
from unittest import TestCase
from libs.pascalVoc import PascalVocWriter, PascalVocReader
from libs.io.yolo import YoloWriter, YoloReader
from libs.scripts.voc_to_yolo import convertAnnotation


class Image(object):
    def __init__(self, h, w, c):
        self._height = h
        self._width = w
        self.colors = c

    def isGrayscale(self):
        return False if self.colors > 1 else True

    def height(self):
        return self._height

    def width(self):
        return self._width


class TestIO(TestCase):

    def testPascalVocRW(self):

        # Test Write/Read
        writer = PascalVocWriter('tests', 'test', (512, 512, 1))
        difficult = 1
        person_box = [60, 40, 430, 504]
        face_box = [113, 40, 450, 403]
        writer.addBndBox(person_box, 'person', difficult)
        writer.addBndBox(face_box, 'face', difficult)
        writer.save('tests/test.xml')

        reader = PascalVocReader('tests/test.xml')
        shapes = reader.getShapes()

        personBndBox = shapes[0]
        face = shapes[1]
        self.assertEqual(personBndBox[0], 'person')
        self.assertEqual(personBndBox[1], [(60, 40), (430, 40), (430, 504), (60, 504)])
        self.assertEqual(face[0], 'face')
        self.assertEqual(face[1], [(113, 40), (450, 40), (450, 403), (113, 403)])

    def testYoloRW(self):

        writer = YoloWriter('tests', 'test', (512, 512, 1))
        person_box = [60, 40, 430, 504]
        face_box = [113, 40, 450, 403]
        writer.addBndBox(person_box, 'person', 0)
        writer.addBndBox(face_box, 'face', 0)
        writer.save(['person', 'face'], 'tests/test.yolo')

        image = Image(512, 512, 1)

        reader = YoloReader('tests/test.yolo', image)
        shapes = reader.getShapes()

        personBndBox = shapes[0]
        face = shapes[1]

        self.assertEqual(personBndBox[0], 'person')
        self.assertEqual(personBndBox[1],
                         [(60, 40), (430, 40), (430, 504), (60, 504)])
        self.assertEqual(face[0], 'face')
        # The reason for the 402s is the float math
        self.assertEqual(face[1],
                         [(113, 40), (450, 40), (450, 402), (113, 402)])


