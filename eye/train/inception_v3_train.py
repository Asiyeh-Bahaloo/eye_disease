def train(model, data_loader, batch_size, epochs):
    history = model.fit(x=data_loader,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    )
    return history, model