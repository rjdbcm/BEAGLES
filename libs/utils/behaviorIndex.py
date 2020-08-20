import os
from traces import TimeSeries
from datetime import timedelta
import pandas as pd
import json
from libs.constants import EPOCH


class BehaviorIndex:
    def __init__(self, file_list, start_time=None, measure_interval=None):
        """
        Takes a list of csv files and spits out a behavior index
        """
        if start_time:
            start_minutes, start_seconds = start_time.split(':')
            self.start_time = EPOCH + timedelta(minutes=start_minutes, seconds=start_seconds)
        else:
            self.start_time = EPOCH
        try:
            interval_minutes, interval_seconds = measure_interval.split(':')
            self.interval = self.start_time + timedelta(minutes=interval_minutes, seconds=interval_seconds)
        except AttributeError:
            pass
        self.series = list()
        for file in file_list:
            if not os.path.isfile(file):
                raise FileNotFoundError(file)
            ts = TimeSeries.from_csv(file)
            ts.remove_points_from_interval(EPOCH, self.start_time) if start_time else None
            ts.remove_points_from_interval(self.start_time, ts.last_key()) if measure_interval else None
            self.series.append(ts)
        self.file_list = file_list

    def __len__(self):
        return len(self.series)

    # def group_single_beh(self):
    #     """
    #     Returns: dict of group single behavior indices
    #     """
    #     bi = self.individual_single_beh()
    #     sum_ = Counter()
    #     for k, v in bi.items():
    #         if isinstance(v, dict):
    #             sum_ += Counter(v)
    #     return dict(sum_)

    def individuals(self):
        """
        Returns: dict of individual single behavior indices per csv
        """
        beh_indices = list()
        for ts in self.series:
            bi = self.beh_slice(ts)
            beh_indices.append(bi)
        return dict(zip(self.file_list, beh_indices))

    # def group_total_beh(self):
    #     """
    #     Returns: float of group total behavior index as a fraction of interval
    #     """
    #     if len(self.series) > 1:
    #         ts = TimeSeries()
    #         ts = ts.merge(self.series)
    #         return self.beh_interval(ts)
    #     else:
    #         pass

    def total_intervals(self):
        """
        Returns: dict of form {file_name: total_beh_interval)
        """
        intervals = list()
        for ts in self.series:
            bi = self.beh_interval(ts)
            intervals.append(bi)
        return dict(zip(self.file_list, intervals))

    def report(self):
        intervals = self.total_intervals()
        report = self.individuals()
        for indv, behs in report.items():
            behs.update({'interval': intervals[indv]})
        return report

    def to_dataframe(self):
        report = self.report()
        return pd.concat({k: pd.DataFrame(v, index=pd.RangeIndex(0,1)).T for k, v in report.items()}, axis=0)

    def json_report(self, destination=None):
        report = self.report()
        report = json.dumps(report)
        return report

    @staticmethod
    def beh_slice(ts):
        d = dict()
        for k, v in ts.distribution().items():
            d.update({k: v})
        return d

    # make ts slices for individual behaviors
    @staticmethod
    def beh_interval(ts):
        delta = ts.last_key() - ts.first_key()
        delta = delta.total_seconds()
        return delta


