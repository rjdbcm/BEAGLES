import time
import math
import os
from multiprocessing.pool import ThreadPool
import numpy as np
from libs.backend.net.tfnet import TFNet


class Predictor(TFNet):
    def __init__(self, flags):
        super(Predictor, self).__init__(flags=flags)
        self.pool = ThreadPool()

    def predict(self):
        def speak_total_time(last, inp_feed):
            self.logger.info(f'Total time = {last}s / {len(inp_feed)} inps = {len(inp_feed) / last} ips')

        self.flags = self.read_flags()
        inp_path = self.flags.imgdir
        all_inps = os.listdir(inp_path)
        all_inps = [i for i in all_inps if self.framework.is_inp(i)]
        if not all_inps:
            exit(f'Error: Failed to find any images in {inp_path}')

        batch = min(self.flags.batch, len(all_inps))

        # predict in batches
        n_batch = int(math.ceil(len(all_inps) / batch))
        for j in range(n_batch):
            self.logger.info(range(n_batch))
            from_idx = j * batch
            to_idx = min(from_idx + batch, len(all_inps))

            # collect images input in the batch
            this_batch = all_inps[from_idx:to_idx]
            inp_feed = self.pool.map(lambda inp: (
                np.expand_dims(self.framework.preprocess(
                    os.path.join(inp_path, inp)), 0)), this_batch)

            # Feed to the net
            feed_dict = {self.inp: np.concatenate(inp_feed, 0)}
            self.logger.info(f'Forwarding {len(inp_feed)} inputs ...')
            start = time.time()
            out = self.sess.run(self.out, feed_dict)
            stop = time.time()
            last = stop - start
            speak_total_time(last, inp_feed)

            # Post processing
            self.logger.info(f'Post processing {len(inp_feed)} inputs ...')
            start = time.time()
            self.pool.map(lambda p: (lambda i, prediction:
                          self.framework.postprocess(
                            prediction,
                            os.path.join(inp_path, this_batch[i])))(*p),
                          enumerate(out))
            stop = time.time()
            last = stop - start
            speak_total_time(last, inp_feed)