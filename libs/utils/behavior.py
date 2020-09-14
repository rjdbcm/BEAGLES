import os
import csv
import sys
import json
from datetime import timedelta
from traces import TimeSeries, plot


class BehaviorAnalysis:
    """
    Takes a list of header-less csv files and analyzes. Extends traces.TimeSeries to work
    with datetime.timedelta instead of datetime.datetime.
    """
    def __init__(self, file_list, classes: list, start_time: timedelta, measure_interval: timedelta, ordinal=False, **kwargs):
        self.start_time = start_time
        self.measure_interval = measure_interval
        self.end_time = self.start_time + measure_interval
        self._classes = classes
        self._ordinal = ordinal
        self.series = list()
        kwargs.update({'value_transform': lambda cls: self.order.get(cls)}) if ordinal else None
        kwargs.update({'time_transform': lambda t: timedelta(seconds=float(t))})
        for file in file_list:
            if not os.path.isfile(file):
                print(f'File {file} not found. Skipping...')
                continue
            try:
                ts = TimeSeries.from_csv(file, **kwargs)
                ts = self._trim(ts)
            except StopIteration:
                ts = TimeSeries()
            self.series.append(ts)
        self.file_list = file_list
        self._data = self.report()

    def __call__(self):
        return self.data

    @property
    def data(self):
        return self._data

    @property
    def classes(self):
        return self._classes

    @property
    def ordinal(self):
        return self._ordinal

    @property
    def order(self):
        return {v: i for i, v in enumerate(self.classes)} if self.ordinal else None

    def __len__(self):
        return len(self.series)

    def save_plot(self, **kwargs):
        for ts in self.series:
            plt, _ = plot.plot(ts, **kwargs)
            plt.show()

    def _trim(self, ts):
        before = [time for time in ts._d.keys() if time < self.start_time]
        after = [time for time in ts._d.keys() if time > self.end_time]
        [print(i) for i in after]
        unused = before + after
        [ts.remove(time) for time in unused] if unused else None
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
        beh_indices = [val / self.measure_interval.total_seconds() for val in values]
        return dict(zip(self.file_list, beh_indices))

    def report(self):
        intervals = self.total_intervals()
        bouts = self.total_bouts()
        report = self.individuals()
        for indv, behs in report.items():
            behs.update({'beh_interval': intervals[indv]})
            behs.update({'beh_index': intervals[indv] / self.measure_interval.total_seconds()})
            behs.update({'total_bouts': bouts[indv]})
        return report

    def json_report(self, **kwargs):
        report = self.data
        report = json.dumps(report, **kwargs)
        return report

    def csv_report(self, target=sys.stdout):
        report = self.data
        fields = ['file', 'beh_interval', 'beh_index', 'total_bouts'] + self.classes
        w = csv.DictWriter(target, fields)
        w.writeheader()
        for key, val in sorted(report.items()):
            row = {'file': key}
            row.update(val)
            w.writerow(row)

    def beh_indices(self, ts):
        behaviors = dict()
        distribution = ts.distribution().items() if not ts.is_empty() else dict()
        [behaviors.update({beh: val}) for beh, val in distribution]
        [behaviors.update({beh: 0.0}) for beh in self.classes if beh not in behaviors.keys()]
        return behaviors

    @staticmethod
    def beh_interval(ts):
        if ts.is_empty():
            return 0
        delta = ts.last_key() - ts.first_key()
        delta = delta.total_seconds()
        return delta
