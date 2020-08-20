import os
import math
import pickle
from functools import reduce
import tensorflow as tf
from libs.utils.errors import GradientNaN
# noinspection PyUnresolvedReferences
from libs.backend.net.frameworks.vanilla.train import loss
from libs.backend.net.tfnet import TFNet


class Trainer(TFNet):
    def __init__(self, flags):
        super(Trainer, self).__init__(flags)

    def train(self):
        self.io_flags()
        loss_ph = self.framework.placeholders
        profile = list()
        batches = self.framework.shuffle(self.annotation_data)
        loss_op = self.framework.loss
        self.flags = self.read_flags()
        fetches = (self.train_op, loss_op, self.summary_op)
        goal = len(self.annotation_data) * self.flags.epoch
        self.total_steps = goal // self.flags.batch
        step_pad = len(str(self.total_steps))
        batch = self.flags.batch
        ckpt = 1
        args = None

        for i, (x_batch, datum, batch_images) in enumerate(batches):
            feed_dict = {loss_ph[key]: datum[key] for key in loss_ph}
            feed_dict[self.inp] = x_batch
            feed_dict.update(self.feed)
            fetched = self.sess.run(fetches, feed_dict)
            loss = fetched[1]
            # Check for exploding/vanishing gradient
            if math.isnan(loss) or math.isinf(loss):
                self.raise_error(GradientNaN(self.flags))
            step_now = self.flags.load + i + 1
            self.writer.add_summary(fetched[2], step_now)
            profile += [loss, batch_images]
            self.logger.info(f'Step {str(step_now).zfill(step_pad)} '
                             f'- Loss {loss:.4f} - Progress {self.flags.progress:.2f}% '
                             f'- Batch {batch_images}')
            args = [step_now, profile]
            ckpt = (i + 1) % (self.flags.save // self.flags.batch)
            count = i * batch
            self.flags.progress = count / goal * 100
            self.io_flags()
            if not ckpt:
                self._save_ckpt(*args)
        if ckpt:
            self._save_ckpt(*args)

    def _save_ckpt(self, step, loss_profile):
        file = '{}-{}{}'
        model = self.meta['name']

        losses = loss_profile[::2]
        image_sets = loss_profile[1::2]
        sample = self.total_steps // 10
        worst_indices = sorted(range(len(losses)), key=lambda sub: losses[sub])[-sample:]
        worst = [(losses[i], image_sets[i]) for i in worst_indices]
        self.logger.info(worst)
        difficult = file.format(model, step, '.difficult')
        difficult = os.path.join(self.flags.backup, difficult)
        with open(difficult, 'a') as difficult_images:
            difficult_images.writelines([f'{str(i)}\n' for i in worst])

        profile = file.format(model, step, '.profile')
        profile = os.path.join(self.flags.backup, profile)
        with open(profile, 'wb') as profile_ckpt:
            pickle.dump(losses, profile_ckpt)

        ckpt = file.format(model, step, '')
        ckpt = os.path.join(self.flags.backup, ckpt)
        self.logger.info('Checkpoint at step {}'.format(step))
        self.saver.save(self.sess, ckpt)


