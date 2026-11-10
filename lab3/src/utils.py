import gc

import tensorflow as tf


def clear_model_training_session() -> None:
    tf.keras.backend.clear_session()
    gc.collect()
