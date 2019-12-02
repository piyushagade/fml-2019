from modules import imports as _

if __name__== "__main__":
    args = _.sys.argv

    print("-" * 100)
    if not "silent" in args:
        print("Cuda: " + ("Available" if _.t.cuda.is_available() else "Unavailable"))
        print("PyTorch: ", _.t.__version__)
    print("")
    
    # Load data
    bad_train_data_object = _.d.load(_, "./data/raw/old_train_data.pkl")
    train_data_object = _.d.load(_, _.g.TRAINING_DATA_PATH)
    raw_train_labels = _.d.load(_, _.g.TRAINING_LABELS_PATH)
    raw_train_data = []
    
    # Preprocess (standardize and pad) and convert bool image data to int
    raw_train_data = _.d.standardize(_, train_data_object)

    # Check if train-validation splits files exists
    if (not "train" in args and not "metrics" in args) and _.os.path.exists(_.g.DATA_STORE_PATH + 'derived/tts_train_data.npy'):
        # Load saved train-validation data splits
        train_data = _.np.load(_.g.DATA_STORE_PATH + 'derived/tts_train_data.npy', allow_pickle=True)
        train_labels = _.np.load(_.g.DATA_STORE_PATH + 'derived/tts_train_labels.npy', allow_pickle=True)
        valid_data = _.np.load(_.g.DATA_STORE_PATH + 'derived/tts_valid_data.npy', allow_pickle=True)
        valid_labels = _.np.load(_.g.DATA_STORE_PATH + 'derived/tts_valid_labels.npy', allow_pickle=True)
    elif not "train" in args and not "metrics" in args:
        print("\nThis seems to be the first run. Please train the model to generate all the files necessary to use other features of this program.\n")
        _.sys.exit(0)

    # Do stuff
    if "display" in args:
        # Display all characters in the dataset
        _.d.show_all_characters(_, train_data_object, 18, 30)

    elif "experiments-phase-1" in args:
        # Phase 1
        parameters_dictionary = dict(
            learning_rate = _.g.LEARNING_RATE_LIST,
            batch_size = _.g.BATCH_SIZE_LIST
        )
        parameters = [v for v in parameters_dictionary.values()]

        for learning_rate, batch_size in _.product(*parameters):
            print("\nHyperparameters: ", learning_rate, batch_size)
            _.cnn.train(_, train_data, train_labels, valid_data, valid_labels, batch_size=batch_size, learning_rate=learning_rate, epochs=15)

    elif "experiments-phase-2" in args:
        # Phase 2
        parameters_dictionary = dict(
            kernel_size = _.g.KERNEL_SIZE_LIST,
            dropout_prob = _.g.DROPOUT_PROB_LIST,
        )
        parameters = [v for v in parameters_dictionary.values()]

        for kernel_size, dropout_prob in _.product(*parameters):
            _.g.KERNEL_SIZE = kernel_size
            _.g.DROPOUT_PROB = dropout_prob
            print("\nHyperparameters: ", kernel_size, dropout_prob)
            _.cnn.train(_, train_data, train_labels, valid_data, valid_labels, batch_size=_.g.BATCH_SIZE, learning_rate=_.g.LEARNING_RATE, epochs=15)

    elif "train" in args:
        
        # Split data in train and validation sets and create new data sets
        train_data, valid_data, train_labels, valid_labels = _.kcv.split(_, raw_train_data, raw_train_labels, _.g.TRAIN_TEST_SPLIT)
        
        # Make the labels start from 0 rather than 1
        for i, data in enumerate(train_labels):
            train_labels[i] = data - 1
        for i, data in enumerate(valid_labels):
            valid_labels[i] = data - 1

        # Save new datesets and replace the old ones
        _.np.save(_.g.DATA_STORE_PATH + 'derived/tts_train_data.npy', train_data)
        _.np.save(_.g.DATA_STORE_PATH + 'derived/tts_train_labels.npy', train_labels)
        _.np.save(_.g.DATA_STORE_PATH + 'derived/tts_valid_data.npy', valid_data)
        _.np.save(_.g.DATA_STORE_PATH + 'derived/tts_valid_labels.npy', valid_labels)
        
        # Train CNN model
        model, train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list, valid_true_labels, valid_predicted_labels = _.cnn.train(_, train_data, train_labels, valid_data, valid_labels, batch_size=_.g.BATCH_SIZE, learning_rate=_.g.LEARNING_RATE, epochs=_.g.NUM_EPOCHS)

        # Save the model and plot
        identifier = str(_.g.LEARNING_RATE).replace(".", "") + "_" + str(_.g.BATCH_SIZE).replace(".", "") + "_"  + str(_.g.KERNEL_SIZE).replace(".", "") + "_" + str(_.g.DROPOUT_PROB).replace(".", "") + "_"
        _.t.save(model, _.g.MODEL_STORE_PATH + identifier + 'conv_net_model.ckpt')
        _.t.save(train_loss_list, _.g.METRICS_SAVE_PATH + identifier + 'train_loss.ckpt')
        _.t.save(train_accuracy_list, _.g.METRICS_SAVE_PATH + identifier + 'train_accuracy.ckpt')
        _.t.save(valid_loss_list, _.g.METRICS_SAVE_PATH + identifier + 'valid_loss.ckpt')
        _.t.save(valid_accuracy_list, _.g.METRICS_SAVE_PATH + identifier + 'valid_accuracy.ckpt')
        _.t.save(valid_true_labels, _.g.METRICS_SAVE_PATH + identifier + 'valid_true_labels.ckpt')
        _.t.save(valid_predicted_labels, _.g.METRICS_SAVE_PATH + identifier + 'valid_predicted_labels.ckpt')

        print("\nNew model saved: './models/" + _.g.MODEL_STORE_PATH + identifier + 'conv_net_model.ckpt')
        print("\nMetrics saved in: './metrics/'")


    elif "metrics" in args:
        identifier = str(_.g.LEARNING_RATE).replace(".", "") + "_" + str(_.g.BATCH_SIZE).replace(".", "") + "_"  + str(_.g.KERNEL_SIZE).replace(".", "") + "_" + str(_.g.DROPOUT_PROB).replace(".", "") + "_"
        train_loss = _.t.load(_.g.METRICS_SAVE_PATH + identifier + 'train_loss.ckpt')
        train_accuracy = _.t.load(_.g.METRICS_SAVE_PATH + identifier + 'train_accuracy.ckpt')
        valid_loss = _.t.load(_.g.METRICS_SAVE_PATH + identifier + 'valid_loss.ckpt')
        valid_accuracy = _.t.load(_.g.METRICS_SAVE_PATH + identifier + 'valid_accuracy.ckpt')
        valid_true_labels = _.t.load(_.g.METRICS_SAVE_PATH + identifier + 'valid_true_labels.ckpt')
        valid_predicted_labels = _.t.load(_.g.METRICS_SAVE_PATH + identifier + 'valid_predicted_labels.ckpt')

        # Plot loss
        _.plt.plot([t for t in range(0, len(train_loss), 1)], train_loss, '-', label='Training loss', color="red")
        _.plt.plot([t for t in range(0, len(valid_loss), 1)], valid_loss, '-', label='Validation loss', color="blue")
        _.plt.legend([" Training loss", " Validation loss"], ncol=2, loc='upper center', 
            bbox_to_anchor=[0.5, 1.1], 
            columnspacing=1.0, labelspacing=0.0,
            handletextpad=0.0, handlelength=1.5,
            fancybox=True, shadow=True)
        _.plt.savefig(_.g.METRICS_SAVE_PATH + "loss.png",facecolor='w', edgecolor='w', pad_inches=0.1)
        _.plt.show()

        # Plot accuracy
        _.plt.plot([t for t in range(0, len(train_accuracy), 1)], train_accuracy, label='Training accuracy', color="red")
        _.plt.plot([t for t in range(0, len(valid_accuracy), 1)], valid_accuracy, label='Validation accuracy', color="blue")
        _.plt.legend([" Training accuracy", " Validation accuracy"], ncol=2, loc='upper center', 
            bbox_to_anchor=[0.5, 1.1], 
            columnspacing=1.0, labelspacing=0.0,
            handletextpad=0.0, handlelength=1.5,
            fancybox=True, shadow=True)
        _.plt.savefig(_.g.METRICS_SAVE_PATH + "accuracy.png",facecolor='w', edgecolor='w', pad_inches=0.1)
        _.plt.show()

        # Plot confusion matrix
        from sklearn.metrics import confusion_matrix
        _.plt.matshow(confusion_matrix(valid_true_labels, valid_predicted_labels, labels=_.g.MAP_CLASSES))
        _.plt.colorbar()
        _.plt.savefig(_.g.METRICS_SAVE_PATH + "confusion_matrix.png", facecolor='w', edgecolor='w', pad_inches=0.1)
        _.plt.ylabel("Actual")
        _.plt.xlabel("Predicted")
        _.plt.show()


        # Show classification report
        from sklearn import metrics
        print(metrics.classification_report(valid_true_labels, valid_predicted_labels))


        print("PNGs saved to: './metrics/'.\n")


    elif "predict" in args:

        # Test on an image
        bad_train_data_object = _.d.standardize(_, bad_train_data_object)
        _.cnn.predict(_, bad_train_data_object[1609])

    elif "test" in args:

        test_data_file_name = args[args.index("test") + 1]

        # Load the test data
        test_data_object = _.d.load(_, test_data_file_name)
        
        # Transform ndarray to int32 dtype from int64
        try:
            test_data_object = test_data_object.astype('int32') * 255
        except:
            pass

        # Standardize data
        test_data_object = _.d.standardize(_, test_data_object)

        # Test CNN model
        predicted_labels = _.cnn.test(_, test_data_object, dev=_.g.DEV)

