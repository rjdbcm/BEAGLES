
class BoundingBox(object):
    def __init__(self):
        self.boxlist = []

    def addBndBox(self, bounding_box, name, difficult):
        bndbox = {'xmin': bounding_box[0],
                  'ymin': bounding_box[1],
                  'xmax': bounding_box[2],
                  'ymax': bounding_box[3]}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)