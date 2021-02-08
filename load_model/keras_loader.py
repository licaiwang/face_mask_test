import tensorflow as tf
from tensorflow.keras.models import model_from_json

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def load_keras_model(json_path, weight_path):
    model = model_from_json(open("models/face_mask_detection.json").read())
    model.load_weights("models/face_mask_detection.hdf5")
    return model


def keras_inference(model, img_arr):
    result = model.predict(img_arr)
    y_bboxes = result[0]
    y_scores = result[1]
    return y_bboxes, y_scores
