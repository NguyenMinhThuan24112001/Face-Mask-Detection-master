from FaceDetection import faceDetection
from FaceMask import faceMask
import cv2
import imutils
import time
from imutils.video import VideoStream
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

class detectMask:
    def __init__(self):
        self.faceDetection = faceDetection()
        self.faceMask = faceMask()
    def detectMaskVideo(self):
        vs = VideoStream(src=0).start()
        time.sleep(2.0)


        while True:

            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            detections = self.faceDetection.faceDetection(frame)
            locs, faces = self.faceDetection.processDetection(frame, detections)
            preds = self.faceMask.faceMask(faces)


            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred


                label = "Mask" if mask > 0.8 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                conf = mask if mask > 0.8 else withoutMask

                label = "{}: {:.2f}%".format(label, conf * 100)


                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


            cv2.imshow("Frame", imutils.resize(frame, height=800))
            key = cv2.waitKey(1) & 0xFF


            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.stop()

    def detectMaskImage(self, filename):
        image = cv2.imread(f"{filename}")
        (h, w) = image.shape[:2]
        detections = self.faceDetection.faceDetection(image)
        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.6:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")


                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))


                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)


                (mask, withoutMask) = self.faceMask.maskNet.predict(face)[0]


                label = "Mask" if mask > 0.8 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                conf = mask if mask > 0.8 else withoutMask

                label = "{}: {:.2f}%".format(label, conf * 100)


                cv2.putText(image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)


        return label, image

