import os
import tensorflow as tf
import math
import pickle
from beagles.base.errors import GradientNaN
# noinspection PyUnresolvedReferences
from beagles.backend.net.frameworks.vanilla.train import loss
from beagles.backend.net.tfnet import TFNet


class Trainer(tf.Module):
    def __init__(self, flags):
        super(Trainer, self).__init__()
        self.net = TFNet(flags)


    def __call__(self):
        self.net.io_flags()
        loss_ph = self.net.framework.placeholders
        profile = list()
        batches = self.net.framework.shuffle(self.net.annotation_data)
        loss_op = self.net.framework.loss
        self.flags = self.net.io.read_flags()
        fetches = (self.net.train_op, loss_op, self.net.summary_op)
        goal = len(self.net.annotation_data) * self.flags.epoch
        total_steps = goal // self.flags.batch
        step_pad = len(str(total_steps))
        batch = self.flags.batch
        ckpt = 1
        args = None

        for i, (x_batch, datum) in enumerate(batches):
            feed_dict = {loss_ph[key]: datum[key] for key in loss_ph}
            feed_dict[self.net.inp] = x_batch
            feed_dict.update(self.net.feed)
            fetched = self.net.sess.run(fetches, feed_dict)
            loss_val = fetched[1]
            # Check for exploding/vanishing gradient
            if math.isnan(loss_val) or math.isinf(loss_val):
                self.net.raise_error(GradientNaN(self.flags))
            step_now = self.flags.load + i + 1
            self.net.writer.add_summary(fetched[2], step_now)
            profile += loss_val

            step = str(step_now).zfill(step_pad)
            prg = self.flags.progress
            self.net.logger.info(f'Step {step} - Loss {loss_val:.4f} - Progress {prg:.2f}%')

            args = [step_now, profile]
            ckpt = (i + 1) % (self.flags.save // self.flags.batch)
            count = i * batch
            self.flags.progress = count / goal * 100
            self.net.io_flags()
            if not ckpt:
                self._save_ckpt(*args)
        if ckpt:
            self._save_ckpt(*args)

    def _save_ckpt(self, step, loss_profile):
        file = '{}-{}{}'
        model = self.net.meta['name']

        profile = file.format(model, step, '.profile')
        profile = os.path.join(self.flags.backup, profile)
        with open(profile, 'wb') as profile_ckpt:
            pickle.dump(loss_profile, profile_ckpt)

        ckpt = file.format(model, step, '')
        ckpt = os.path.join(self.flags.backup, ckpt)
        self.net.logger.info('Checkpoint at step {}'.format(step))
        self.net.saver.save(self.net.sess, ckpt)


