from tensorflow.keras.losses import binary_crossentropy 

def eval(model, data_loader, batch_size):
    model.evaluate(x = data_loader, batch_size= batch_size)
