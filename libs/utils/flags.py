import sys
import json


class Flags(dict):
    """
    Allows you to set and get {key: value} pairs like attributes.
    This allows compatibility with argparse.Namespace objects.
    """

    def __init__(self, defaults=True):
        super(Flags, self).__init__()
        if defaults:
            self.get_defaults()

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def get_defaults(self):


    def from_json(self, file):
        data = dict(json.load(file))
        for attr, value in data.items():
            self.__setattr__(attr, value)
        return self

    def to_json(self, file=sys.stdout):
        return json.dump(dict(self.items()), fp=file)
