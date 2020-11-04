from beagles.backend.darknet.layer import Layer

class conv_lstm_layer(Layer):
    def setup(self, size, stride, pad, peephole, batch_norm, activation):
        self.size = size
        self.stride = stride
        self.pad = pad
        self.peephole = bool(peephole)
        self.batch_norm = bool(batch_norm)

    def finalize(self, *args):
        """Not Implemented"""
        pass

class lstm_layer(Layer):
    def setup(self, num_cells, batch_norm):
        self.batch_norm = bool(batch_norm)
        self.num_cells = num_cells

    def finalize(self, *args):
        """Not Implemented"""
        pass

class rnn_layer(Layer):
    def setup(self, num_cells, batch_norm,  activation):
        self.batch_norm = bool(batch_norm)
        self.num_cells = num_cells
        self.activation = activation

    def finalize(self, *args):
        """Not Implemented"""
        pass

class gru_layer(Layer):
    def setup(self, num_cells, batch_norm):
        self.batch_norm = bool(batch_norm)
        self.num_cells = num_cells

    def finalize(self, *args):
        """Not Implemented"""
        pass
