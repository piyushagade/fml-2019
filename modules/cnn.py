# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
# https://datascience.stackexchange.com/questions/45916/loading-own-train-data-and-labels-in-dataloader-using-pytorch
# https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

def train(_imports, train_data, train_labels, valid_data, valid_labels, batch_size=32, learning_rate=0.0001, epochs=50):
    _ = _imports

    train_data = _.Tensor(train_data).to(_.DEVICE)
    train_labels = _.Tensor(train_labels).to(_.DEVICE)

    valid_data = _.Tensor(valid_data).to(_.DEVICE)
    valid_labels = _.Tensor(valid_labels).to(_.DEVICE)

    # Concatenate train data and labels
    train_dataset = _.TensorDataset(train_data, train_labels)
    valid_dataset = _.TensorDataset(valid_data, valid_labels)

    # Convert ndarray to tensors
    train_loader = _.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = _.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    comment = f' bs={batch_size} lr={learning_rate} ks={_.g.KERNEL_SIZE} s={_.g.STRIDE} ks={_.g.DROPOUT_PROB}'
    tb = _.SummaryWriter(comment=comment)
    
    # Model
    model = _.m.ConvNet_1().to(_.DEVICE)

    # Loss and optimizer
    criterion = _.nn.CrossEntropyLoss()
    optimizer = _.t.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []

    train_loss_list = []
    train_accuracy_list = []
    valid_loss_list = []
    valid_accuracy_list = []
    valid_true_labels = []
    valid_predicted_labels = []

    for epoch in range(epochs):

        train_total_loss = 0.0
        train_total_correct = 0
        train_total_predicted = 0

        valid_total_loss = 0.0
        valid_total_correct = 0
        valid_total_predicted = 0

        # Train
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images.unsqueeze(1))
            loss = criterion(outputs, labels.long())
            loss_list.append(loss.item())
            train_total_loss += loss.item()

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()   # Clear previous gradients
            loss.backward()         # Compute gradients of all variables wrt loss
            optimizer.step()        # Perform updates using calculated gradients
            
            # Track the accuracy
            _var, predicted = _.t.max(outputs.data, 1)
            is_prediction_correct = (predicted == labels.view(-1).long()).sum().item()
            train_total_correct += (predicted == labels.view(-1).long()).sum().item()
            train_total_predicted += labels.size(0)

            
        train_loss_list.append(train_total_loss / len(train_loader))
        train_accuracy_list.append(train_total_correct / train_total_predicted)

        # Validate
        model.eval()
        with _.t.no_grad():
            for i, (images, labels) in enumerate(valid_loader):

                outputs = model(images.unsqueeze(1))
                _var, predicted = _.t.max(outputs.data, 1)
                is_prediction_correct = (predicted == labels.view(-1).long()).sum().item()
                valid_total_correct += (predicted == labels.view(-1).long()).sum().item()
                valid_total_predicted += labels.size(0)
                
                for j in range(0, labels.size(0)):
                    valid_true_labels.append(_.g.MAP_CLASSES[labels.view(-1).long()[j].item() + 1])
                    valid_predicted_labels.append(_.g.MAP_CLASSES[predicted[j].item() + 1])

                # Run the forward pass
                outputs = model(images.unsqueeze(1))

                # Get loss for this batch
                loss = criterion(outputs, labels.long())
                valid_total_loss += loss.item()

        valid_loss_list.append(valid_total_loss / len(valid_loader))
        valid_accuracy_list.append(valid_total_correct / valid_total_predicted)

        print("\nEpoch " + str(epoch) + ":") 
        print("\t" + "Training:    Loss: " + str(round(train_total_loss / len(train_loader), 3)) + ", " + "Accuracy: " + str(round(train_total_correct / train_total_predicted, 3)))
        print("\t" + "Validation:  Loss: " + str(round(valid_total_loss / len(valid_loader), 3)) + ", " + "Accuracy: " + str(round(valid_total_correct / valid_total_predicted, 3)))
        
        tb.add_scalar("Training Loss", train_total_loss / len(train_loader), epoch)
        tb.add_scalar("Training Accuracy", train_total_correct / train_total_predicted, epoch)
        tb.add_scalar("Validation Loss", valid_total_loss / len(valid_loader), epoch)
        tb.add_scalar("Validation Accuracy", valid_total_correct / valid_total_predicted, epoch)

    tb.close()

    return model, train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list, valid_true_labels, valid_predicted_labels

def validate(_imports, validation_data, validation_labels, batch_size=32, learning_rate=0.0001, model=None): 
    _ = _imports
    if model == None:
        print("'model' parameter required for validation.")
        return None, None, None

    # Make the labels start from 0 rather than 1
    for i, data in enumerate(validation_labels):
        validation_labels[i] = data - 1

    validation_data = _.Tensor(validation_data)
    validation_labels = _.Tensor(validation_labels)

    # Concatenate train data and labels
    dataset = _.TensorDataset(validation_data, validation_labels)

    test_loader = _.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    identifier = str(_.g.LEARNING_RATE).replace(".", "") + "_" + str(_.g.BATCH_SIZE).replace(".", "") + "_"  + str(_.g.KERNEL_SIZE).replace(".", "") + "_" + str(_.g.DROPOUT_PROB).replace(".", "") + "_"
    model = _.t.load(_.g.MODEL_STORE_PATH + identifier + 'conv_net_model.ckpt')

    # Test the model
    model.eval()
    with _.t.no_grad():
        correct = 0
        total = 0
        acc_list = []
        for images, labels in test_loader:
            outputs = model(images.unsqueeze(1))
            _var, predicted = _.t.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1).long()).sum().item()
            acc_list.append(correct / total)

        print('Test Accuracy of the model on the validation images: {} %'.format((correct / total) * 100))

    
    return model, acc_list

def predict(_imports, image):
    _ = _imports

    # Convert image list to tensor
    image = _.t.FloatTensor(image)
    image = image.to(_.DEVICE)
    
    # Load model
    identifier = str(_.g.LEARNING_RATE).replace(".", "") + "_" + str(_.g.BATCH_SIZE).replace(".", "") + "_"  + str(_.g.KERNEL_SIZE).replace(".", "") + "_" + str(_.g.DROPOUT_PROB).replace(".", "") + "_"
    model = _.t.load(_.g.MODEL_STORE_PATH + identifier + 'conv_net_model.ckpt')
    model = model.to(_.DEVICE)


    # Test the model
    model.eval()
    with _.t.no_grad():
        correct = 0
        total = 0
        output = model(image.unsqueeze(0).unsqueeze(0))
        _var, predicted = _.t.max(output.data, 1)
        
        print("Neurons' max: ", _.t.max(output.data, 1)[0].item())
        print(output.data)

        # Check for unknown data
        if _.t.max(output.data, 1)[0].item() <= _.g.UNKNOWN_CLASS_THRESHOLD:
            _.d.show_character(_, image.to("cpu"), label="Prediction: " + _.g.MAP_LABELS[0])
        else:
            _.d.show_character(_, image.to("cpu"), label="Prediction: " + _.g.MAP_LABELS[predicted.item() + 1])

def test(_imports, test_data, model=None, dev=False): 
    _ = _imports

    # Load model if no model provided in arguments
    if model == None:
        identifier = str(_.g.LEARNING_RATE).replace(".", "") + "_" + str(_.g.BATCH_SIZE).replace(".", "") + "_"  + str(_.g.KERNEL_SIZE).replace(".", "") + "_" + str(_.g.DROPOUT_PROB).replace(".", "") + "_"
        model = _.t.load(_.g.MODEL_STORE_PATH + identifier + 'conv_net_model.ckpt')

    # Test the model
    model.eval()
    with _.t.no_grad():
        predictions_labels = []
        predictions_characters = []
        for image in _.Tensor(test_data).to(_.DEVICE):
            output = model(image.unsqueeze(0).unsqueeze(0))
            _var, predicted = _.t.max(output.data, 1)

            # Check for unknown data
            if _.t.max(output.data, 1)[0].item() <= _.g.UNKNOWN_CLASS_THRESHOLD:
                predictions_labels.append(-1)
                predictions_characters.append(_.g.MAP_LABELS[0])
                if dev: 
                    print("\nPredicted" + _.g.MAP_LABELS[predicted.item() + 1])
                    print(output.data)
                    _.d.show_character(_, image.to("cpu"), label="Prediction: " + _.g.MAP_LABELS[0])
            else:
                predictions_labels.append(predicted.item() + 1)
                predictions_characters.append(_.g.MAP_LABELS[predicted.item() + 1])
                if dev: 
                    print("\n" + _.g.MAP_LABELS[predicted.item() + 1])
                    print(output.data)
                    _.d.show_character(_, image.to("cpu"), label="Prediction: " + _.g.MAP_LABELS[predicted.item() + 1])

            
            
    return predictions_labels, predictions_characters
