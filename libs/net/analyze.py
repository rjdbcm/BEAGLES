import os
from ..utils.postprocess import BehaviorIndex


def analyze(self, file_list):
    def _writer(analysis_file, items):
        with open(analysis_file, mode='a') as file:
            file.write(items[0])
            file.write(items[1])
        return

    bi = BehaviorIndex(file_list)
    if len(file_list) > 1:
        for i in file_list:
            analysis_file = os.path.splitext(i)[0] + '_analysis.json'
            _writer(analysis_file,
                    bi.individual_behs())
        analysis_file = 'group_analysis.json'
        _writer(analysis_file, bi.group_behs())
    else:
        analysis_file = os.path.splitext(file_list[0])[0] + '_analysis.json'
        _writer(analysis_file, bi.individual_behs())

