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

    def __getstate__(self):
        pass

    def get_defaults(self):
        with open('resources/flags.json', 'r') as f:
            for attr, value in dict(json.load(f)).items():
                self.__setattr__(attr, value)
