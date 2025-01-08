from tensorflow import keras

# TODO: Should be very easy to replace with torch code
def intermediateOutputs(model, batch, layer=None, mode=None):
    """
    Fuction to get intermadiate feature vectors

    Parameters
    ----------
    batch : tf.Tensor
            A batch of data
    layer : int, optional
            The layer for which to get intermediate features
    mode : str, optional
            'pre' to create a pre-model which takes in the input data and gives out intermediate activations,
                    'post' to take in intermediate activations and give out the model predictions

    Returns
    -------
    tf.keras.Model()
            An extractor model
    """

    model_ = keras.Sequential()
    model_.add(keras.Input(shape=(batch[0][0].shape)))
    for layer_ in model.layers:
        model_.add(layer_)

    if layer is not None and mode == "pre":
        if layer >= 0:
            extractor = keras.Model(
                inputs=model.layers[0].input,
                outputs=model.layers[layer].output,
            )
        else:

            extractor = keras.Model(
                inputs=model.layers[0].input,
                outputs=model.layers[0].input,
            )
    elif layer is not None and mode == "post":
        input_ = keras.Input(shape=(model.layers[layer].input.shape[1:]))
        next_layer = input_
        for layer in model.layers[layer : layer + 2]:
            next_layer = layer(next_layer)
        extractor = keras.Model(input_, next_layer)
    else:
        extractor = keras.Model(
            inputs=model.layers[0].input,
            outputs=[layer.output for layer in model.layers],
        )

    return extractor
