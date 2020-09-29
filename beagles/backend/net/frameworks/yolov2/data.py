def batch(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's 
    input & loss layer correspond to this chunk
    """
    # sizes are passed to avoid having duplicate get_feed_values methods for
    # YOLO and YOLOv2
    H, W, _ = self.meta['out_size']
    return self.get_feed_values(chunk, H, W)


