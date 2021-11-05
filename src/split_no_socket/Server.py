from torch import nn, optim


class Server:
    # Init Server with client_ids, top_model
    def __init__(self, client_ids, top_model):
        self.client_ids = client_ids
        self.top_model = top_model
        self.optimizer = optim.SGD(self.top_model.parameters(), lr=0.003, momentum=0.9)
        self.init_param = self.top_model.state_dict()
        self.update_cache = {}

    # Receive vector2Server, labels from Client
    # Do ForwardProp, BackProp for top_model
    # Return gradient to Client
    def top_calculation(self, vector2Server, labels):
        # Cleaning gradients
        self.optimizer.zero_grad()

        # ForwardProp of top_model
        output = self.top_model(vector2Server)

        # Calculate losses
        criterion = nn.NLLLoss()
        loss = criterion(output, labels)

        # BackProp of top_model
        loss.backward()

        # Optimize weights
        self.optimizer.step()

        # Return gradient to Client
        return vector2Server.grad, loss.item()

    # Init setting before an iteration
    def init_setting(self):
        self.init_param = self.top_model.state_dict()
        self.update_cache = {}

    # Record the updated parameters of top_model by this Client's data
    # Reset top_model to the init parameter of this iteration(for next Client to train)
    def record_update(self, client_id):
        # Record the updated parameter of top_model by this Client's data
        updated_param = self.top_model.state_dict()
        if client_id in self.update_cache.keys():
            print('Duplicate training on a Client in this iteration!')
            exit(-1)
        else:
            self.update_cache[client_id] = updated_param

        # Reset top_model to the init parameter of this iteration(for next Client to train)
        self.top_model.load_state_dict(self.init_param)

    # Aggression for top_model
    def top_aggression(self):
        num = len(self.client_ids)
        if num == 0:
            return

        # Calculate the mean of Clients's updated parameters for top_model
        if self.client_ids[0] not in self.update_cache.keys():
            print('Client {} doesn\'t train in this iteration!'.format(self.client_ids[0]))
            exit(-1)
        model_dict = self.update_cache[self.client_ids[0]]

        for i in range(1, num):
            client_id = self.client_ids[i]
            if client_id not in self.update_cache.keys():
                print('Client {} doesn\'t train in this iteration!'.format(client_id))
                exit(-1)
            else:
                client_dict = self.update_cache[client_id]
                for weights in model_dict.keys():
                    model_dict[weights] = model_dict[weights] + client_dict[weights]
        for weights in model_dict.keys():
            model_dict[weights] = model_dict[weights] / num

        # Use load_state_dict() to update the parameter of top_model
        self.top_model.load_state_dict(model_dict)
