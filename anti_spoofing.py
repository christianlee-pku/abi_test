import cv2
import onnxruntime as ort
import numpy as np
import os

def increased_crop(img, bbox : tuple, bbox_inc : float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]
    
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img


def make_prediction(rgb_frame, bbox, anti_spoof):

    if len(bbox) < 4:
        return None
    
    bbox = list(bbox)

    y1, x2, y2, x1 = bbox
    bbox = (x1, y1, x2, y2)

    pred = anti_spoof([increased_crop(rgb_frame, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)   
    
    return bbox, label, score


# onnx model
class AntiSpoof:
    def __init__(self,
                 weights: str = None,
                 model_img_size: int = 128):
        super().__init__()
        self.weights = weights
        self.model_img_size = model_img_size
        self.ort_session, self.input_name = self._init_session_(self.weights)

    def _init_session_(self, onnx_model_path: str):
        ort_session = None
        input_name = None
        if os.path.isfile(onnx_model_path):
            try:
                ort_session = ort.InferenceSession(onnx_model_path, 
                                                   providers=['CUDAExecutionProvider'])
            except:
                ort_session = ort.InferenceSession(onnx_model_path, 
                                                   providers=['CPUExecutionProvider']) 
            input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name

    def preprocessing(self, img): 
        new_size = self.model_img_size
        old_size = img.shape[:2] # old_size is in (height, width) format

        ratio = float(new_size)/max(old_size)
        scaled_shape = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        img = cv2.resize(img, (scaled_shape[1], scaled_shape[0]))

        delta_w = new_size - scaled_shape[1]
        delta_h = new_size - scaled_shape[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def postprocessing(self, prediction):
        softmax = lambda x: np.exp(x)/np.sum(np.exp(x))
        pred = softmax(prediction)
        return pred
        #return np.argmax(pred)

    def __call__(self, imgs : list):
        if not self.ort_session:
            return False

        preds = []
        for img in imgs:
            onnx_result = self.ort_session.run([],
                {self.input_name: self.preprocessing(img)})
            pred = onnx_result[0]
            pred = self.postprocessing(pred)
            preds.append(pred)
        return preds