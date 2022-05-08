from tensorflow.keras.models import load_model
import numpy as np


class faceMask:
    def __init__(self):
        self.maskNet = load_model("mask_detector.model")
        print(type(self.maskNet))

    def faceMask(self,faces):
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(faces, batch_size=32)

            return preds


