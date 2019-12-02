from modules import imports as _

# Usage:
#   python test.py model_file_name dataset_file_name [dataset_labels_file_name] [True|False]
#
# Second argument:
#   Model's file name
#
# Optional third argument:
#   The third argument is optional. If you specify the labels file, a classification report 
#   will be generated.
#
# Optional fourth argument:
#   The fourth argument is optional. Setting it to True standardizes/preprocesses the images to 
#   ensure the images are all 60x60 in size, and does other operations. We recommend it to be set 
#   to True. Default value is also True.
#   
# Example:
#   python ./models/model_a_b_aug_3_layer.ckpt ./data/test/test_data.npy ./data/test/test_labels.npy

if __name__== "__main__":
    args = _.sys.argv

    if len(args) == 1:
        print("\nPlease refer to README for usage.")
    elif len(args) >= 2:
        model_file_name = args[1]
        test_data_file_name = args[2]
        test_labels_object = None

        if len(args) == 4:
            # Load actual test labels
            test_labels_file_name = args[3]
            test_labels_object = _.d.load(_, test_labels_file_name)

        # Load the test data
        test_data_object = _.d.load(_, test_data_file_name)

        # Transform ndarray to int32 dtype from int64
        try:
            test_data_object = test_data_object.astype('int32') * 255
        except:
            pass
        
        # Standardize data by default or if explicitly asked for
        if (len(args) == 5 and args[4] == True) or len(args) <= 4:
            test_data_object = _.d.standardize(_, test_data_object, resize=True)
        else:
            print("\nSkipping standardizing data.")
            test_data_object = _.d.standardize(_, test_data_object, resize=False)

        # Load model
        model = _.t.load(model_file_name)

        # Test CNN model
        predicted_labels, predicted_characters = _.cnn.test(_, test_data_object, model=model, dev=_.g.DEV or True)

        # Write to files
        _.np.save(_.g.RESULTS_SAVE_PATH + "predicted_labels.npy", predicted_labels) 
        _.np.save(_.g.RESULTS_SAVE_PATH + "predicted_characters.npy", predicted_characters)
        
        if len(args) == 3:
            # Show classification report
            from sklearn import metrics
            print(metrics.classification_report(test_labels_object, predicted_labels))
        
        print("\nPredictions saved to: './results/' as .npy files.\n")
    