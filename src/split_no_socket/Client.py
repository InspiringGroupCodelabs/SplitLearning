import torch
from torch import nn, optim
from torch.autograd import Variable

class Client:
    # Init Client with client_id, train_set, val_set, bottom_model
    def __init__(self, client_id, train_set, val_set, bottom_model):
        self.client_id = client_id
        self.train_set = train_set
        self.val_set = val_set
        self.bottom_model = bottom_model
        self.optimizer = optim.SGD(self.bottom_model.parameters(), lr=0.003, momentum=0.9)

    # Use this Client's data to train the whole model in a iteration
    def train(self, Server, epochs):
        # Load this Client's data for train
        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=64, shuffle=True)

        for epoch in range(epochs):
            running_loss = 0
            img = 0
            for images, labels in train_loader:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)

                # Cleaning gradients
                self.optimizer.zero_grad()

                # ForwardProp of bottom_model
                output = self.bottom_model(images)

                # Prepare vector for Server
                vector2Server = Variable(output.data, requires_grad=True)

                # Get gradient from Server
                gradient2Client, loss = Server.top_calculation(vector2Server, labels)

                # BackProp of bottom_model
                output.backward(gradient2Client)

                # Optimize the weights
                self.optimizer.step()

                running_loss += loss
                img = img + 1
                print("Client {} Epoch {} Batch {} - Training loss: {}".format(self.client_id, epoch, img, running_loss / len(train_loader)))
            else:
                print("Client {} Epoch {} - Training loss: {}".format(self.client_id, epoch, running_loss / len(train_loader)))

            # Call the Server to record the updated parameters of top_model by this Client's data
            Server.record_update(self.client_id)

    def val(self, top_model):
        # Load the data for val
        val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=64, shuffle=True)

        correct_count, all_count = 0, 0
        for images, labels in val_loader:
            for i in range(len(labels)):
                # Flatten MNIST images into a 784 long vector
                img = images[i].view(1, 784)

                with torch.no_grad():
                    # ForwardProp of bottom_model
                    output1 = self.bottom_model(img)
                    y2 = Variable(output1.data, requires_grad=False)
                    # ForwardProp of top_model
                    logps = top_model(y2)

                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if (true_label == pred_label):
                    correct_count += 1
                all_count += 1

        print("\nNumber Of Images Tested =", all_count)
        print("Client {} Model Accuracy =".format(self.client_id), (correct_count / all_count), "\n")


# Aggression for Clients's bottom_model(Not Secure)
def bottom_aggression(clients):
    num = len(clients)
    if num == 0:
        return

    # Use state_dict() to get the parameter of Clients's bottom_model
    # Calculate the mean of these parameters
    model_dict = clients[0].bottom_model.state_dict()
    for i in range(1, num):
        client = clients[i]
        client_dict = client.bottom_model.state_dict()
        for weights in model_dict.keys():
            model_dict[weights] = model_dict[weights] + client_dict[weights]
    for weights in model_dict.keys():
        model_dict[weights] = model_dict[weights] / num

    # Use load_state_dict() to update the parameter of Clients's bottom_model
    for client in clients:
        client.bottom_model.load_state_dict(model_dict)