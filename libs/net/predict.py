import time
import math
import os
from multiprocessing.pool import ThreadPool
import numpy as np

pool = ThreadPool()


def predict(self):
    def _speak_total_time(last, inp_feed):
        self.logger.info('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

    self.flags = self.read_flags()
    inp_path = self.flags.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.Framework.is_inp(i)]
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
            np.expand_dims(self.Framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp: np.concatenate(inp_feed, 0)}
        self.logger.info('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time()
        last = stop - start
        _speak_total_time(last, inp_feed)

        # Post processing
        self.logger.info(
            'Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
                            self.Framework.postprocess(
                                prediction,
                                os.path.join(inp_path, this_batch[i])))(*p),
                 enumerate(out))
        stop = time.time()
        last = stop - start
        _speak_total_time(last, inp_feed)