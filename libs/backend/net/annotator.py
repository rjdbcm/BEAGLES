from datetime import datetime
import os
import csv
import numpy as np
import cv2
from imutils.video import FileVideoStream
from libs.constants import EPOCH
from libs.backend.net.tfnet import TFNet
from libs.io.flags import FlagIO


class Annotator:
    def __init__(self, flags):
        self.flags = flags
        self.net = TFNet(flags)

    def annotate(self):
        input_video = self.flags.video
        frame_count = 0
        for i in input_video:
            fvs = FileVideoStream(i)
            total_frames = int(fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            annotation_file = f'{os.path.splitext(i)[0]}_annotations.csv'

            if os.path.exists(annotation_file):
                self.net.logger.info("Overwriting existing annotations")
                os.remove(annotation_file)

            self.net.logger.info(f'Annotating {input_video}')

            while fvs.more():
                frame_count += 1
                frame = fvs.read()
                self.flags.progress = round((100 * frame_count / total_frames), 0)
                if frame_count % 10 == 0:
                    self.net.io_flags()
                frame = np.asarray(frame)
                result = self.return_predict(frame)

                # This is a hackish way of making sure we can quantify videos
                # taken at different times
                epoch = EPOCH.timestamp()
                time_elapsed = fvs.stream.get(cv2.CAP_PROP_POS_MSEC) / 1000
                self.write_annotations(annotation_file, result, time_elapsed, epoch)
                if self.flags.kill:
                    break
            else:
                break
            # When everything done, release the capture
            fvs.stream.release()

    def draw_box(self, original_img, predictions):
        """
        Args:
            original_img: A numpy ndarray
            predictions: A nested dictionary object of the form
                        {"label": str, "confidence": float,
                        "topleft": {"x": int, "y": int},
                        "bottomright": {"x": int, "y": int}}
        Returns:
            A numpy ndarray with boxed detections
        """
        new_image = np.copy(original_img)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        for result in predictions:

            confidence = result.max_prob
            top_x = result.left
            top_y = result.top
            btm_x = result.right
            btm_y = result.btm

            header = " ".join([result.mess, str(round(confidence, 3))])

            if confidence > self.flags.threshold:
                new_image = cv2.rectangle(new_image, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
                new_image = cv2.putText(new_image, header, (top_x, top_y - 5), font, 0.8, (0, 230, 0), 1, cv2.LINE_AA)

        return new_image

    def write_annotations(self, annotation_file, prediction, time_elapsed, epoch):

        def _center(x1, y1, x2, y2):
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            return x, y

        with open(annotation_file, mode='a') as file:
            file_writer = csv.writer(file, delimiter=',',
                                     quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
            for result in prediction:
                if result.max_prob > self.flags.threshold:

                    center_x, center_y = _center(result.left,
                                                 result.top,
                                                 result.right,
                                                 result.btm)

                    file_writer.writerow([datetime.fromtimestamp(epoch + time_elapsed),
                                         result.mess, result.max_prob, center_x, center_y,
                                         result.left, result.top, result.right, result.btm])

    def return_predict(self, im):
        assert isinstance(im, np.ndarray), 'Image is not a np.ndarray'
        h, w, _ = im.shape
        im = self.net.framework.resize_input(im)
        this_inp = np.expand_dims(im, 0)
        feed_dict = {self.net.inp: this_inp}

        out = self.net.sess.run(self.net.out, feed_dict)[0]
        boxes = self.net.framework.findboxes(out)
        threshold = self.flags.threshold
        boxesInfo = list()
        for box in boxes:
            processed_box = self.net.framework.process_box(box, h, w, threshold)
            if processed_box is None:
                continue
            boxesInfo.append(processed_box)
        return boxesInfo
