import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
import pickle
from multiprocessing.pool import ThreadPool

train_stats = (
    'Training statistics - '
    'Learning rate: {} '
    'Batch size: {}    '
    'Epoch number: {}  '
    'Backup every: {}  '
)
pool = ThreadPool()


def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.flags.backup, profile)
    with open(profile, 'wb') as profile_ckpt:
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.flags.backup, ckpt)
    self.logger.info('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    self.io_flags()
    loss_ph = self.framework.placeholders
    loss_mva = None
    profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss
    for i, (x_batch, datum) in enumerate(batches):
        self.flags = self.read_flags()
        if not i:
            self.logger.info(train_stats.format(
                self.flags.lr, self.flags.batch,
                self.flags.epoch, self.flags.save
            ))
            count = 0

        feed_dict = {
            loss_ph[key]: datum[key]
            for key in loss_ph}
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.flags.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if math.isnan(loss):
            if self.flags.clip:
                self.flags.error = "Looks like the neural net lost the gradient. Try restarting from your last " \
                                   "checkpoint."
            if not self.flags.clip:
                self.flags.error = "Looks like the neural net lost the gradient. Try restarting from the last " \
                                   "checkpoint. If this keeps happening try using the --clip argument when you " \
                                   "restart from the last checkpoint."

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.flags.load + i + 1

        count += self.flags.batch
        goal = self.flags.size * self.flags.epoch
        self.flags.progress = count / goal * 100
        self.io_flags()

        if self.flags.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.logger.info(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i + 1) % (self.flags.save // self.flags.batch)
        args = [step_now, profile]

        if not ckpt:
            _save_ckpt(self, *args)

    if ckpt:
        _save_ckpt(self, *args)


def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
        'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp: this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.flags.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo


def predict(self):
    self.flags = self.read_flags()
    inp_path = self.flags.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.flags.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        self.logger.info(range(n_batch))
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp: np.concatenate(inp_feed, 0)}
        self.logger.info('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time()
        last = stop - start
        self.logger.info('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.logger.info('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
                            self.framework.postprocess(
                                prediction, os.path.join(inp_path, this_batch[i])))(*p),
                 enumerate(out))
        stop = time.time()
        last = stop - start

        # Timing
        self.logger.info('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
