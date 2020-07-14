from traces import TimeSeries
from collections import Counter


class BehaviorIndex:
    def __init__(self, file_list):
        """
        Takes a list of csv files and spits out a behavior index
        """
        self.ts_list = list()
        self._dict = dict()
        for file in file_list:
            ts = TimeSeries.from_csv(file)
            self.ts_list.append(ts)
        self.file_list = file_list

    def group_single_beh(self):
        """
        Returns: dict of group single behavior indices
        """
        bi = self.individual_single_beh()
        sum_ = Counter()
        for k, v in bi.items():
            if isinstance(v, dict):
                sum_ += Counter(v)
        return dict(sum_)

    def individual_single_beh(self):
        """
        Returns: dict of individual single behavior indices per csv
        """
        l = list()
        for ts in self.ts_list:
            bi = self.beh_slice(ts)
            l.append(bi)
        return dict(zip(self.file_list, l))

    def group_total_beh(self):
        """
        Returns: float of group total behavior index as a fraction of interval
        """
        if len(self.ts_list) > 1:
            ts = TimeSeries()
            ts = ts.merge(self.ts_list)
            return self.beh_index(ts)
        else:
            return self.individual_total_beh()

    def individual_total_beh(self):
        """
        Returns: dict of form {file_name: total_beh_index)
        """
        l = list()
        for ts in self.ts_list:
            bi = self.beh_index(ts)
            l.append(bi)
        return dict(zip(self.file_list, l))

    def individual_behs(self):
        return [self.individual_total_beh(), self.individual_single_beh()]

    def group_behs(self):
        return [self.group_total_beh(), self.group_single_beh()]

    def beh_slice(self, ts):
        d = dict()
        for k, v in ts.distribution().items():
            d.update({k: v * self.beh_index(ts)})
        return d

    # make ts slices for individual behaviors
    def beh_index(self, ts):
        delta = ts.last_key() - ts.first_key()
        delta = delta.total_seconds()
        return delta
