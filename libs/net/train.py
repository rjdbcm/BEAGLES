import os
import math
import pickle
import tensorflow as tf
from libs.utils.errors import GradientNaN
from libs.utils.cyclic_learning_rate import cyclic_learning_rate

train_stats = (
    'Training statistics - '
    'Learning rate: {} '
    'Batch size: {}    '
    'Epoch number: {}  '
    'Backup every: {}  '
)


def train(self):
    self.io_flags()
    loss_ph = self.framework.placeholders
    loss_mva = None
    profile = list()
    goal = None
    total_steps = None
    step_pad = None
    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        self.flags = self.read_flags()
        feed_dict = {
            loss_ph[key]: datum[key]
            for key in loss_ph}
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)
        fetches = [self.train_op, loss_op]
        if self.flags.summary:
            fetches.append(self.summary_op)

        # Start the session
        try:
            fetched = self.sess.run(fetches, feed_dict)
        except tf.errors.OpError as oe:
            if oe.error_code == 3 and "nan" in oe.message.lower():
                self.raise_error(GradientNaN(self.flags), tf_traceback=oe)
            raise
        loss = fetched[1]

        # single shot calculations
        if not i:
            self.logger.info(train_stats.format(
                self.flags.lr, self.flags.batch,
                self.flags.epoch, self.flags.save
            ))
            count = 0
        if not goal:
            goal = self.flags.size * self.flags.epoch
        if not total_steps:
            total_steps = goal // self.flags.batch
            step_pad = len(str(total_steps))
        if not loss_mva:
            loss_mva = loss

        # Check for exploding/vanishing gradient
        if math.isnan(loss) or math.isinf(loss):
            self.raise_error(GradientNaN(self.flags))

        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.flags.load + i + 1

        # assign_op = global_step.assign(step_now)
        # self.sess.run(assign_op)

        # Calculate and send progress
        # noinspection PyUnboundLocalVariable
        count += self.flags.batch
        self.flags.progress = count / goal * 100
        self.io_flags()

        if self.flags.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {} - progress {}'
        self.logger.info(
            form.format(str(step_now).zfill(step_pad),
                        format(loss, '.14f'),
                        format(loss_mva, '.14f'),
                        "{:=6.2f}%".format(self.flags.progress)))
        profile += [(loss, loss_mva)]

        ckpt = (i + 1) % (self.flags.save // self.flags.batch)
        args = [step_now, profile]

        if not ckpt:
            self._save_ckpt(*args)

    # noinspection PyUnboundLocalVariable
    if ckpt:
        # noinspection PyUnboundLocalVariable
        self._save_ckpt(*args)


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


