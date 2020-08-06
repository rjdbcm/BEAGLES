from datetime import datetime
import os
import csv
import numpy as np
import cv2


def annotate(self):
    INPUT_VIDEO = self.flags.fbf
    FRAME_NUMBER = 0
    cap = cv2.VideoCapture(INPUT_VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    annotation_file = os.path.splitext(INPUT_VIDEO)[0] + '_annotations.csv'

    if os.path.exists(annotation_file):
        self.logger.info("Overwriting existing annotations")
        os.remove(annotation_file)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    max_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    max_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(os.path.splitext(INPUT_VIDEO)[0] + '_annotated.avi', fourcc,
                          fps, (int(max_x), int(max_y)))
    self.logger.info('Annotating ' + INPUT_VIDEO)

    while True:
        FRAME_NUMBER += 1
        ret, frame = cap.read()
        if ret:
            self.flags.progress = round((100 * FRAME_NUMBER / total_frames), 0)
            if FRAME_NUMBER % 10 == 0:
                self.io_flags()
            frame = np.asarray(frame)
            result = self.return_predict(frame)
            new_frame = self.draw_box(frame, result)

            # This is a hackish way of making sure we can quantify videos
            # taken at different times
            epoch = datetime(1970, 1, 1, 0, 0).timestamp()
            time_elapsed = cap.get(cv2.CAP_PROP_POS_MSEC)
            self.write_annotations(annotation_file, result, time_elapsed, epoch)
            out.write(new_frame)
            if self.flags.kill:
                break
        else:
            break
    # When everything done, release the capture
    out.release()


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
    assert isinstance(im, np.ndarray), \
        'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp: this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.flags.threshold
    boxesInfo = list()
    for box in boxes:
        processed_box = self.framework.process_box(box, h, w, threshold)
        if processed_box is None:
            continue
        boxesInfo.append(processed_box)
    return boxesInfo
