import os
import io
import csv
import sys
import json
from datetime import datetime
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, AutoLocator
from typing import List, AnyStr
from datetime import timedelta
from traces import TimeSeries
from matplotlib import font_manager
from beagles.io.obs import datetime_from_filename, DATETIME_FORMAT

COLUMNS = {
    'time': 0,
    'beh': 1,
    'prob': 2,
    'center_x': 3,
    'center_y': 4,
    'left': 5,
    'top': 6,
    'right': 7,
    'bottom': 8
}


class BehaviorAnalysis:
    """
    Create behavior analyses from unevenly-spaced timeseries annotation data.
    """
    def __init__(self, classes: List[AnyStr], start_time: int = 0,
                 measure_interval: int = 600, decay_time: float = 1.0, ordinal=False):
        self.decay_time = decay_time
        self.start_time = start_time
        self.measure_interval = measure_interval
        self.end_time = self.start_time + self.measure_interval
        self._ordinal = ordinal
        self._classes = classes
        self.metadata: List[pd.DataFrame] = list()
        self.series: List[TimeSeries] = list()
        self.file_list = list()
        self._data = dict()

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
        return {cls: i for i, cls in enumerate(self.classes)}

    @property
    def rev_order(self):
        return {i: cls for i, cls in enumerate(self.classes)}

    @property
    def individual_behs(self):
        """
        Returns: dict of individual single behavior indices per csv
        """
        indices = [self._beh_indices(ts) for ts in self.series]
        return dict(zip(self.file_list, indices))

    @property
    def total_intervals(self):
        """
        Returns: dict of form {file_name: total_beh_interval)
        """
        intervals = [self._beh_interval(ts) for ts in self.series]
        return dict(zip(self.file_list, intervals))

    @property
    def total_bouts(self):
        bouts = list()
        for ts in self.series:
            try:
                bout = len(list(ts.iterperiods()))
                bouts.append(bout)
            except KeyError:
                bouts.append(0)
        return dict(zip(self.file_list, bouts))

    @property
    def total_behavior_index(self):
        values = self.total_intervals.values()
        beh_indices = [val / self.measure_interval for val in values]
        return dict(zip(self.file_list, beh_indices))

    def _report(self):
        intervals = self.total_intervals
        bouts = self.total_bouts
        report = self.individual_behs
        for indv, behs in report.items():
            behs.update({'beh_interval': intervals[indv]})
            behs.update({'beh_index': intervals[indv] / self.measure_interval})
            behs.update({'total_bouts': bouts[indv]})
        return report

    def _beh_indices(self, ts):
        behaviors = dict()
        distribution = ts.distribution().items() if not ts.is_empty() else dict()
        if self.ordinal:
            [behaviors.update({self.rev_order.get(beh): val}) for beh, val in distribution]
            [behaviors.update({cls: 0.0}) for cls in self.order.keys() if cls not in behaviors.keys()]
        else:
            [behaviors.update({beh: val}) for beh, val in distribution]
            [behaviors.update({beh: 0.0}) for beh in self.classes if beh not in behaviors.keys()]
        return behaviors

    def _beh_interval(self, ts: TimeSeries):
        if ts.is_empty():
            return 0
        return sum([en-st for st, en, _ in ts.iterperiods() if en-st < self.decay_time])

    # METHODS #
    def add_annotation(self, file: os.PathLike, **kwargs):
        self.file_list.append(file)

        # setup default kwargs #
        if self.ordinal:
            kwargs.update({'value_transform': lambda cls: self.order.get(cls)})
        if not kwargs.get('time_transform', None):
            kwargs.update({'time_transform': lambda t: float(t)})
        if not kwargs.get('skip_header', None):
            kwargs.update({'skip_header': False})
        if not os.path.isfile(file):
            raise FileNotFoundError(f'File {file} not found.')
        if os.path.getsize(file):
            ts = TimeSeries.from_csv(file, **kwargs).slice(self.start_time, self.end_time)
            meta = pd.read_csv(file, names=COLUMNS.keys())
            meta.drop(meta[meta['time'] < self.start_time].index)
            meta.drop(meta[meta['time'] > self.end_time].index)
        else:
            ts = TimeSeries()
            meta = pd.DataFrame()
        self.metadata.append(meta)
        self.series.append(ts)
        self._data = self._report()

    def rem_annotation(self, file: os.PathLike):
        files_removed = list()
        series_removed = list()
        idx = self.file_list.index(file)
        files_removed.append(self.file_list.pop(idx))
        series_removed.append(self.series.pop(idx))
        self._data = self._report()
        return dict(zip(files_removed, series_removed))

    def to_json(self, **kwargs):
        report = self.data
        report = json.dumps(report, **kwargs)
        return report

    def to_markdown(self):
        md = list()
        header = str()
        header_line = str()
        row = u''
        head = u'|{}'
        end = u'|'
        line = head.format('--')
        report = self.data
        first = True
        for file, data in report.items():
            if first:
                header += head.format('file')
                header_line += line
            row += head.format(os.path.basename(file))
            for k, v in data.items():
                if first:
                    header += head.format(k)
                    header_line += line
                row += head.format(str(v)[:4])
            if first:
                header += end
                header_line += end
            row += end
            info = '\n'.join([row])
            first = False
            md.append(info)
            row = str()
        header = '\n'.join([header, header_line])

        return header, md

    def to_csv(self, target=sys.stdout):
        report = self.data
        fields = ['file', 'beh_interval', 'beh_index', 'total_bouts'] + self.classes
        w = csv.DictWriter(target, fields)
        w.writeheader()
        for key, val in sorted(report.items()):
            row = {'file': key}
            row.update(val)
            try:
                w.writerow(row)
            except ValueError:
                pass

    @property
    def experiment_names(self):
        names = [datetime_from_filename(i).strftime(DATETIME_FORMAT['underscore']) for i in self.file_list]
        return list(dict.fromkeys(names))

    def write_text_summary(self):
        writer = tf.summary.create_file_writer('./data/summaries/')
        writer.set_as_default()
        for name in self.experiment_names:
            idx = [i for i, fname in enumerate(self.file_list) if name in fname]
            header, table = self.to_markdown()
            table = [table[i] for i in idx]
            tf.summary.text(name, tf.constant('\n'.join([header, *table]),
                                              dtype=tf.string), step=0)
            writer.flush()

    def write_image_summary(self, **kwargs):
        def pad(x, y=0.02):
            return x + x*y

        def plot_to_image(figure):
            """Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call."""
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image

        writer = tf.summary.create_file_writer('./data/summaries/')
        for i, ts in enumerate(self.series):
            if ts.is_empty():
                print(f'Skipping empty annotation {self.file_list[i]}')
                continue
            pl, ax = plot(ts, **kwargs)
            pl.set_size_inches(11.0, 4.2)
            ax.set_adjustable("box")
            file = os.path.basename(self.file_list[i])
            ax.set_title(file)
            ax.set_ylabel('Behaviors')
            ytop = pad(len(self.classes)-1)
            yticks = [*[float(i) for i, _ in enumerate(self.classes)], ytop]
            ax.set_yticks(yticks)
            ax.set_yticklabels(self.classes)
            ax.set_xlabel('Time in Seconds')
            xticks = [*[float(i)*60.0 for i in range(self.end_time // 60)], self.end_time]
            ax.set_xticks(xticks)
            ax.set_xticklabels([*[x - self.start_time for x in xticks], 600])
            ax.set_xlim(self.start_time, pad(self.end_time, 0.002))
            with writer.as_default():
                name = datetime_from_filename(self.file_list[i]).strftime(DATETIME_FORMAT['underscore'])
                name = f'{name}/{file}'
                head, table = self.to_markdown()
                md = '/n'.join([head, table[i]])
                tf.summary.image(name, plot_to_image(pl), max_outputs=1, step=0, description=md)
        writer.close()



MIN_ASPECT_RATIO = 1 / 15
MAX_ASPECT_RATIO = 1 / 3
MAX_ASPECT_POINTS = 10

FONTS = [
    ".SF Compact Rounded",
    "Helvetica Neue",
    "Segoe UI",
    "Helvetica",
    "Arial",
    None,
]


def plot(ts, figure_width=12, linewidth=1, marker=".", color="mediumvioletred", aspect_ratio=None, font=None):
    if font is None:
        available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
        for font in FONTS:
            if font in available_fonts:
                break

    if aspect_ratio is None:
        try:
            n_unique_values = len(ts.distribution())
        except KeyError:
            n_unique_values = 0
        scaled = min(MAX_ASPECT_POINTS, max(2, n_unique_values) - 2)
        aspect_ratio = MIN_ASPECT_RATIO + (MAX_ASPECT_RATIO - MIN_ASPECT_RATIO) * (scaled / MAX_ASPECT_POINTS)

    with plt.style.context('seaborn'):
        fig, ax = plt.subplots(figsize=(figure_width, aspect_ratio * figure_width))
        items = ts.items()
        x, y = zip(*items) if items else ([], [])
        ax.scatter(x, y, linewidth=linewidth, marker=marker, color=color)
        ax.set_aspect(75)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(AutoLocator())
        if font:
            plt.xticks(fontname=font)
            plt.yticks(fontname=font)

    return fig, ax

