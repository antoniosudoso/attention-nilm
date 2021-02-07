import tensorflow as tf
from matplotlib import pyplot as plt
from data_generator import DataGenerator
from model import build_model
from utils import *

if __name__ == '__main__':

    # Read the NILM dataset
    main_train, appliance_train = np.array([]), np.array([])
    main_val, appliance_val = np.array([]), np.array([])
    main_test, appliance_test = np.array([]), np.array([])

    # Choose the appliance-specific window size
    window_size = 512

    # Threshold of 15 Watt for detecting the ON/OFF states
    THRESHOLD = 15

    # Build ON/OFF appliance vector for the classification subtask
    appliance_train_classification = np.copy(appliance_train)
    appliance_train_classification[appliance_train_classification <= THRESHOLD] = 0
    appliance_train_classification[appliance_train_classification > THRESHOLD] = 1

    appliance_val_classification = np.copy(appliance_val)
    appliance_val_classification[appliance_val_classification <= THRESHOLD] = 0
    appliance_val_classification[appliance_val_classification > THRESHOLD] = 1

    # Standardization of the main power and normalization of appliance power
    appliance_min_power = np.min(appliance_train)
    appliance_max_power = np.max(appliance_train)
    main_std = np.std(main_train)
    main_mean = np.mean(main_train)

    main_train = standardize_data(main_train, np.mean(main_train), np.std(main_train))
    main_val = standardize_data(main_val, np.mean(main_val), np.std(main_val))

    appliance_train_regression = np.copy(appliance_train)
    appliance_train_regression = normalize_data(appliance_train_regression, appliance_min_power, appliance_max_power)

    appliance_val_regression = np.copy(appliance_val)
    appliance_val_regression = normalize_data(appliance_val_regression, appliance_min_power, appliance_max_power)

    # Dataset generator
    batch_size = 32
    train_generator = DataGenerator(main_train, appliance_train_regression,
                                    appliance_train_classification, window_size, batch_size)
    val_generator = DataGenerator(main_val, appliance_val_regression,
                                  appliance_val_classification, window_size, batch_size)

    train_steps = train_generator.__len__()
    validation_steps = val_generator.__len__()

    # Tune the appliance-dependent parameters
    filters = 32
    kernel_size = 4
    units = 128

    model, att_model = build_model(window_size, filters, kernel_size, units)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(x=train_generator, epochs=100, steps_per_epoch=train_steps,
                        validation_data=val_generator, validation_steps=validation_steps,
                        callbacks=[early_stop], verbose=1)

    # Plotting the results of training
    history_dict = history.history
    plt.title('Loss during training')
    plt.plot(np.arange(len(history.epoch)), history_dict['loss'])
    plt.plot(np.arange(len(history.epoch)), history_dict['val_loss'])
    plt.legend(['train', 'val'])
    plt.show()

    # Test
    appliance_test_classification = np.copy(appliance_test)
    appliance_test_classification[appliance_test_classification <= THRESHOLD] = 0
    appliance_test_classification[appliance_test_classification > THRESHOLD] = 1

    appliance_min_power = np.min(appliance_train)
    appliance_max_power = np.max(appliance_train)

    main_test = standardize_data(main_test, np.mean(main_test), np.std(main_test))

    appliance_test_regression = np.copy(appliance_test)
    appliance_test_regression = normalize_data(appliance_test_regression, appliance_min_power, appliance_max_power)

    batch_size = 32

    test_generator = DataGenerator(main_test, appliance_test_regression,
                                   appliance_test_classification, window_size, batch_size)

    test_steps = test_generator.__len__()

    results = model.evaluate(x=test_generator, steps=test_steps)
    predicted_output, predicted_on_off = model.predict(x=test_generator, steps=test_steps)

    predicted_output *= (appliance_max_power - appliance_min_power)
    predicted_output += appliance_min_power
    # Clip negative values to zero
    predicted_output[predicted_output < 0] = 0.0

    prediction = build_overall_sequence(predicted_output)
    prediction_on_off = build_overall_sequence(predicted_on_off)

    # Compute metrics
    N = 1200
    MAE = mae(prediction, appliance_test)
    SAE = sae(prediction, appliance_test, N=N)
    F1 = f1(prediction_on_off, appliance_test_classification)

    print("MAE = {}".format(MAE))
    print("SAE = {}".format(SAE))
    print("F1 = {}".format(F1))

    # Plot the result of the prediction
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(50, 40))
    axes[0].set_title("Real")
    axes[0].plot(np.arange(len(appliance_test)), appliance_test, color='blue')
    axes[1].set_title("Prediction")
    axes[1].plot(np.arange(len(prediction)), prediction, color='orange')
    axes[2].set_title("Real vs prediction")
    axes[2].plot(np.arange(len(appliance_test)), appliance_test, color='blue')
    axes[2].plot(np.arange(len(prediction)), prediction, color='orange')
    axes[3].set_title("Real on off")
    axes[3].plot(np.arange(len(appliance_test_classification)), appliance_test_classification, color='blue')
    axes[4].set_title("Prediction on off")
    axes[4].plot(np.arange(len(prediction_on_off)), prediction_on_off, color='orange')
    axes[5].set_title("Real vs Prediction on off")
    axes[5].plot(np.arange(len(appliance_test_classification)), appliance_test_classification, color='blue')
    axes[5].plot(np.arange(len(prediction_on_off)), prediction_on_off, color='orange')
    fig.tight_layout()
