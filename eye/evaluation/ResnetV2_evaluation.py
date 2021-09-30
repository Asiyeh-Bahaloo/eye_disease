from eye.utils.utils import print_metrics


def resnet_v2_evaluate(x_test, y_test, model, result_path):

    """[Evaluation]

    Parameters
    ----------
    data_path : [str]
        [Address of data]
    model : [object of the resnetV2 class]
        [Model object from the script ResnetV2.py]
    weights_path : [str]
        [The address which will saves our weights]

    result_path  : [str]
        [where to save the results]

    """

    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ": ", value)

    # test a prediction
    test_predictions_baseline = model.predict(x_test)

    # show the final score
    print_metrics(y_test, test_predictions_baseline, threshold=0.5)
