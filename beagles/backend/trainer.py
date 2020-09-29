import os
import math
import pickle
from beagles.base.errors import GradientNaN
# noinspection PyUnresolvedReferences
from beagles.backend.net.frameworks.vanilla.train import loss
from beagles.backend.net.tfnet import TFNet


class Trainer:
    def __init__(self, flags):
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
        self.total_steps = goal // self.flags.batch
        step_pad = len(str(self.total_steps))
        batch = self.flags.batch
        ckpt = 1
        args = None

        for i, (x_batch, datum, batch_images) in enumerate(batches):
            feed_dict = {loss_ph[key]: datum[key] for key in loss_ph}
            feed_dict[self.net.inp] = x_batch
            feed_dict.update(self.net.feed)
            fetched = self.net.sess.run(fetches, feed_dict)
            loss = fetched[1]
            # Check for exploding/vanishing gradient
            if math.isnan(loss) or math.isinf(loss):
                self.net.raise_error(GradientNaN(self.flags))
            step_now = self.flags.load + i + 1
            self.net.writer.add_summary(fetched[2], step_now)
            profile += [loss, batch_images]
            self.net.logger.info(f'Step {str(step_now).zfill(step_pad)} '
                             f'- Loss {loss:.4f} - Progress {self.flags.progress:.2f}% '
                             f'- Batch {batch_images}')
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

        losses = loss_profile[::2]
        image_sets = loss_profile[1::2]
        sample = self.total_steps // 10
        worst_indices = sorted(range(len(losses)), key=lambda sub: losses[sub])[-sample:]
        worst = [(losses[i], image_sets[i]) for i in worst_indices]
        self.net.logger.info(worst)
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
        self.net.logger.info('Checkpoint at step {}'.format(step))
        self.net.saver.save(self.net.sess, ckpt)


