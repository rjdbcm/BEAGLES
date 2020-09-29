import googletrans
import os


class StringsResourceFile(object):
    def __init__(self, path: str) -> None:
        self.file = path
        self.strings = self.read()

    def read(self) -> dict:
        strings = dict()
        with open(self.file, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                key, string = line.split('=')
                item = dict({key: string})
                strings.update(item)
        return strings

    def translate(self, language: str) -> dict:
        translator = googletrans.Translator()
        translated_strings = dict()
        for k, v in self.strings.items():
            translated_strings[k] = translator.translate(v, language).text
        return translated_strings

    def write(self, language: str, locale: str = None) -> None:
        translated_strings = self.translate(language)
        file, ext = os.path.splitext(self.file)
        file = '-'.join(filter(None, [file, language, locale]))
        file = file + ext
        with open(file, 'w') as f:
            for k, v in translated_strings.items():
                line = '='.join([k, v]) + '\n'
                f.write(line)
