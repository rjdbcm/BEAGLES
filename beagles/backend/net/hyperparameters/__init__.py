from beagles.backend.net.hyperparameters.cyclic_learning_rate import *

cyclic_learning_rate = cyclic_learning_rate
"""Applies cyclic learning rate (CLR).
    From the paper:
        Smith, Leslie N. "Cyclical learning
        rates for training neural networks." 2017.
        `Link <https://arxiv.org/pdf/1506.01186.pdf>`_

    This method lets the learning rate cyclically
    vary between reasonable boundary values
    achieving improved classification accuracy and
    often in fewer iterations. This code varies the 
    learning rate linearly between the minimum (learning_rate)
    and the maximum (max_lr).
      
    It returns the cyclic learning rate. It is computed as:
       
       .. code-block:: python
        
            cycle = floor( 1 + global_step / ( 2 * step_size ) )
            x = abs( global_step / step_size – 2 * cycle + 1 )
            clr = learning_rate + ( max_lr – learning_rate ) * max( 0 , 1 - x )
            
    Note:
        When eager execution is enabled, this function returns
        a function which in turn returns the decayed learning
        rate Tensor. This can be useful for changing the learning
        rate value across different invocations of self.optimizer functions.
       
    Polices:
        'triangular':
          Default, linearly increasing then linearly decreasing the
          learning rate at each cycle.
        'triangular2':
          The same as the triangular policy except the learning
          rate difference is cut in half at the end of each cycle.
          This means the learning rate difference drops after each cycle.
        'exp_range':
          The learning rate varies between the minimum and maximum
          boundaries and each boundary value declines by an exponential
          factor of: gamma^global_step.
       
    Example: 'triangular2' mode cyclic learning rate.
        .. code-block:: python
        
            ...
            global_step = tf.Variable(0, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=clr.cyclic_learning_rate(global_step=global_step, mode='triangular2'))
            train_op = self.optimizer.minimize(loss_op, global_step=global_step)
            ...
            with tf.Session() as sess:
                sess.run(init)
                for step in range(1, num_steps+1):
                  assign_op = global_step.assign(step)
                  sess.run(assign_op)
            ...

    Args:
        global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
            Global step to use for the cyclic computation.  Must not be negative.
        learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
            The initial learning rate which is the lower bound
            of the cycle (default = 0.1).
        max_lr:  A scalar. 
            The maximum learning rate boundary.
        step_size: A scalar. The number of iterations in half a cycle.
            The paper suggests step_size = 2-8 x training iterations in epoch.
        gamma: constant in 'exp_range' mode:
            gamma**(global_step)
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
        name: String.  Optional name of the operation.  Defaults to
            'CyclicLearningRate'.
    Returns:
        A scalar `Tensor` of the same type as `learning_rate`.  The cyclic
        learning rate.
    Raises:
        ValueError: if `global_step` is not supplied.
      
"""