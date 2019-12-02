from modules import imports as _

if __name__== "__main__":
    args = _.sys.argv

    if len(args) == 1:
        print("\nUsing dataset provided by the team for training. To use custom training file, pass the path as first argument.\n")
        train_data_file_name = _.g.TRAINING_DATA_PATH
        train_labels_file_name = _.g.TRAINING_LABELS_PATH
        standardize = True
    elif len(args) >= 3:
        print("\nUsing custom dataset for training.\n")
        train_data_file_name = args[1]
        train_labels_file_name = args[2]
        if len(args) == 4:
            standardize = args[3]
        else:
            standardize = True
    else:
        print("\nThis program requires two arguments. Please refer to README.\n")
        _.sys.exit(0)

    # Load the training data
    train_data_object = _.d.load(_, train_data_file_name)
    train_labels_object = _.d.load(_, train_labels_file_name)

    # Standardize data
    if standardize:
        train_data_object = _.d.standardize(_, train_data_object)
    else:
        print("\nSkipping standardizing data.")
    
    # Split data in train and validation sets and create new data sets
    train_data, valid_data, train_labels, valid_labels = _.kcv.split(_, train_data_object, train_labels_object, _.g.TRAIN_TEST_SPLIT)

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
    print("\nTraining model: # of epochs: " + str(_.g.NUM_EPOCHS) + ", Learning rate: " + str(_.g.LEARNING_RATE) + ", Batch size: " + str(_.g.BATCH_SIZE) + ", Kernel size: " + str(_.g.KERNEL_SIZE) + ", Dropout probability: " + str(_.g.DROPOUT_PROB) + "\n")
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
