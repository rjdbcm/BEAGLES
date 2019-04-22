from darkflow.net.build import TFNet
import numpy as np
import cv2


class fbf(object):

    def annotate(FLAGS):

        INPUT_VIDEO = FLAGS.fbf
        FRAME_NUMBER = 1

        tfnet = TFNet(FLAGS)

        cap = cv2.VideoCapture(INPUT_VIDEO)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter('./output.avi', fourcc, 20.0, (int(width), int(height)))

        def boxing(original_img, predictions):
            newImage = np.copy(original_img)

            for result in predictions:
                top_x = result['topleft']['x']
                top_y = result['topleft']['y']

                btm_x = result['bottomright']['x']
                btm_y = result['bottomright']['y']

                confidence = result['confidence']
                label = result['label'] + " " + str(round(confidence, 3))

                if confidence > 0.3:
                    newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
                    newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                           (0, 230, 0), 1, cv2.LINE_AA)

            return newImage

        def annotate(predictions):

            time = []
            labels = []
            conf = []
            top_x = []
            top_y = []
            btm_x = []
            btm_y = []

            for item in predictions:
                time.append('%.3f' % (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                labels.append(item['label'])
                conf.append('%.3f' % item['confidence'])
                top_x.append(item['topleft']['x'])
                top_y.append(item['topleft']['y'])
                btm_x.append(item['bottomright']['x'])
                btm_y.append(item['bottomright']['y'])
                print('{}{}{}{}{}{}{}'.format(time, labels, conf, top_x, top_y, btm_x, btm_y))

        while (True):
            # Capture frame-by-frame

            ret, frame = cap.read()
            print("Frame {}/{} [{}%]".format(FRAME_NUMBER, total_frames, round(100 * FRAME_NUMBER / total_frames, 1)),
                  end='\r')
            FRAME_NUMBER += 1

            if ret == True:
                frame = np.asarray(frame)
                result = tfnet.return_predict(frame)

                new_frame = boxing(frame, result)

                annotate(result)

                # Display the resulting frame
                out.write(new_frame)
                cv2.imshow('frame', new_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # When everything done, release the capture
        cap.release()
        out.release()
        cv2.destroyAllWindows()
