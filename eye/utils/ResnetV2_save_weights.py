import os


def save_weights(model, weights_path):
    print("saving weights")
    model.save(os.path.join(weights_path, "model_weights_resnetv2.h5"))
