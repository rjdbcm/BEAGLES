class DarknetConfigEmpty(Exception):
    """raised when a darknet configuration is empty"""
    def __init__(self, cfg):
        Exception.__init__(self, "Configuration is empty: {}".format(cfg))


class GradientNaN(Exception):
    """Raised in cases of exploding or vanishing gradient"""
    def __init__(self, flags):
        clip = "--clip argument" if flags.cli else "'Clip Gradients' checkbox"
        option = "." if flags.clip else " or turning on gradient clipping" \
                                       " using the {}.".format(clip)
        Exception.__init__(
            self, "Looks like the neural net lost the gradient try restarting"
                  " from the last checkpoint with a lower learning rate{}".format(
                   option))


class VariableIsNone(Exception):
    """Raised when a variable cannot be restored"""
    def __init__(self, var):
        Exception.__init__(self, "Cannot find and load: {}".format(var.name))
