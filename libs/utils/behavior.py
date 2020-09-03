import os
import sys
import csv
from traces import TimeSeries, plot
from datetime import timedelta, datetime
import pandas as pd
import json
from libs.constants import EPOCH


class BehaviorAnalysis:
    """
    Takes a list of csv files and analyzes.
    """
    def __init__(self, file_list, classes: list, start_time=None, measure_interval=None):
        if start_time:
            start_minutes, start_seconds = start_time.split(':')
            self.start_time = EPOCH + timedelta(minutes=int(start_minutes),
                                                seconds=int(start_seconds))
        else:
            self.start_time = EPOCH
        self.measure_interval = measure_interval
        self.end_time = self.start_time + timedelta(seconds=measure_interval)
        self.classes = classes
        self.series = list()
        self.empties = list()
        for file in file_list:
            if not os.path.isfile(file):
                print(f'File {file} not found. Skipping...')
                continue
            try:
                ts = TimeSeries.from_csv(file)
                ts = self.trim(ts)
            except StopIteration:
                ts = TimeSeries()
            self.series.append(ts)
        self.file_list = file_list

    def __len__(self):
        return len(self.series)

    def save_plot(self):
        for ts in self.series:
            plt, _ = plot.plot(ts)
            plt.show()

    def trim(self, ts):
        before_start = list()
        after_end = list()
        for i, time in enumerate(ts._d.keys()):
            if self.start_time > time:
                before_start.append(time)
            elif time > self.end_time:
                after_end.append(time)
        if before_start:
            for time in before_start:
                del ts._d[time]
        if after_end:
            for time in after_end:
                del ts._d[time]
        return ts

    def individuals(self):
        """
        Returns: dict of individual single behavior indices per csv
        """
        indices = [self.beh_indices(ts) for ts in self.series]
        return dict(zip(self.file_list, indices))

    def total_intervals(self):
        """
        Returns: dict of form {file_name: total_beh_interval)
        """
        intervals = [self.beh_interval(ts) for ts in self.series]
        return dict(zip(self.file_list, intervals))

    def total_bouts(self):
        bouts = list()
        for ts in self.series:
            try:
                bout = len(list(ts.iterperiods()))
                bouts.append(bout)
            except KeyError:
                bouts.append(0)
        return dict(zip(self.file_list, bouts))

    def total_behavior_index(self):
        values = self.total_intervals().values()
        beh_indices = [val / self.measure_interval for val in values]
        return dict(zip(self.file_list, beh_indices))

    def report(self):
        intervals = self.total_intervals()
        bouts = self.total_bouts()
        report = self.individuals()
        for indv, behs in report.items():
            behs.update({'beh_interval': intervals[indv]})
            behs.update({'beh_index': intervals[indv] / self.measure_interval})
            behs.update({'total_bouts': bouts[indv]})
        return report

    def json_report(self, **kwargs):
        report = self.report()
        report = json.dumps(report, **kwargs)
        return report

    def csv_report(self, target=sys.stdout):
        report = self.report()
        fields = ['file'] + self.classes + ['beh_interval', 'beh_index', 'total_bouts']
        w = csv.DictWriter(target, fields)
        w.writeheader()
        for key, val in sorted(report.items()):
            row = {'file': key}
            row.update(val)
            w.writerow(row)

    def beh_indices(self, ts):
        behaviors = dict()
        distribution = ts.distribution().items() if not ts.is_empty() else dict()
        for beh, value in distribution:
            behaviors.update({beh: value})
        for beh in self.classes:
            behaviors.update({beh: 0.0}) if beh not in behaviors.keys() else None
        return behaviors

    @staticmethod
    def beh_interval(ts):
        if ts.is_empty():
            return 0
        delta = ts.last_key() - ts.first_key()
        delta = delta.total_seconds()
        return delta


