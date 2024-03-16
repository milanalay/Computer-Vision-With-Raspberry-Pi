import numpy as np
import cv2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import utils


# Model selection and switching
_MODELS = {
    'model1': 'deeplabv3.tflite',
}

# Preprocessing and post-processing
_PREPROCESSING_OPTIONS = {
    'resize': (640, 480),
    'normalization': (0.0, 1.0),
    'data_augmentation': True,
}
_POSTPROCESSING_OPTIONS = {
    'crf': False,  # Conditional Random Fields
    'graph_optimization': False,
}

# Multi-threading and batching
_NUM_THREADS = 4
_BATCH_SIZE = 1



class ImageSegmenter:
    def __init__(self, model_name, num_threads):
        self.model_name = model_name
        self.num_threads = num_threads

        self.base_options = self._create_base_options()
        self.segmentation_options = processor.SegmentationOptions(
            output_type=processor.OutputType.CATEGORY_MASK)
        self.options = vision.ImageSegmenterOptions(
            base_options=self.base_options, segmentation_options=self.segmentation_options)

        self.segmenter = vision.ImageSegmenter.create_from_options(self.options)

    def _create_base_options(self):
        base_options = core.BaseOptions(
            file_name=_MODELS[self.model_name],
            num_threads=self.num_threads)

        return base_options
    
    def segment(self, image):
        tensor_image = vision.TensorImage.create_from_array(image)
        segmentation_result = self.segmenter.segment(tensor_image)
        return segmentation_result
    

# Apply data augmentation techniques
def apply_data_augmentation(image):
    # Random horizontal flip
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)

    # Random rotation
    angle = np.random.uniform(-15, 15)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))

    # Random brightness adjustment
    beta = np.random.uniform(-0.2, 0.2)
    image = cv2.convertScaleAbs(image, beta=beta)

    return image



# Background subtraction variables
_BACKGROUND_SUBTRACTION_ENABLED = True
_BACKGROUND_SUBTRACTOR = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)


def background_subtraction(frame):
    mask = _BACKGROUND_SUBTRACTOR.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    roi = cv2.bitwise_not(mask)
    return roi
