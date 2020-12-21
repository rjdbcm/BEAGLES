from beagles.base.flags import Flags


class DarknetConfigEmpty(Exception):
    """raised when a darknet configuration is empty"""
    def __init__(self, cfg):
        Exception.__init__(self, f"Configuration is empty: {cfg}")


class GradientNaN(Exception):
    """Raised in cases of exploding or vanishing gradient"""
    def __init__(self, flags=None):
        if flags is None:
            flags = Flags()
        clip = "clip command" if flags.cli else "'Clip Gradients' checkbox"
        opt = "." if flags.clip else f" or turning on gradient clipping using the {clip}."
        self.message = f"Looks like the neural net lost the gradient try restarting" \
                       f" from the last checkpoint with a lower learning rate{opt}"
        Exception.__init__(
            self, self.message)


class VariableIsNone(Exception):
    """Raised when a variable cannot be restored"""
    def __init__(self, var):
        Exception.__init__(self, f"Cannot find and load: {var.name}")
