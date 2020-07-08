import os
import unittest
from stringBundle import StringBundle


class TestStringBundle(unittest.TestCase):

    def test_loadDefaultBundle_withoutError(self):
        strBundle = StringBundle.getBundle('en')
        self.assertEqual(strBundle.getString("openDir"), 'Open Folder', 'Fail to load the default bundle')

    def test_fallback_withoutError(self):
        strBundle = StringBundle.getBundle('zh-TW')
        self.assertEqual(strBundle.getString("openDir"), u'\u958B\u555F\u76EE\u9304', 'Fail to load the zh-TW bundle')

    def test_setInvalidLocaleToEnv_printErrorMsg(self):
        try:
            prev_lc = os.environ['LC_ALL']
            prev_lang = os.environ['LANG']
        except KeyError:
            prev_lc = ""
            prev_lang= ""
        os.environ['LC_ALL'] = 'UTF-8'
        os.environ['LANG'] = 'UTF-8'
        strBundle = StringBundle.getBundle()
        self.assertEqual(strBundle.getString("openDir"), 'Open Folder', 'Fail to load the default bundle')
        os.environ['LC_ALL'] = prev_lc
        os.environ['LANG'] = prev_lang


if __name__ == '__main__':
    unittest.main()
