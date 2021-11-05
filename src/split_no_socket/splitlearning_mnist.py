from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch import nn, optim
from split_no_socket.Client import Client, bottom_aggression
from split_no_socket.Server import Server
from copy import deepcopy

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download dataset
train_set = datasets.MNIST('../datasets/MNIST/train', download=True, train=True, transform=transform)
val_set = datasets.MNIST('../datasets/MNIST/val', download=True, train=False, transform=transform)

# Split dataset
train_sets = random_split(train_set, [10000, 10000, 10000, 10000, 10000, 10000])

# Define models
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

bottom_model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
					nn.ReLU(),
					nn.Linear(hidden_sizes[0], hidden_sizes[1]))


top_model = nn.Sequential(nn.ReLU(),
					nn.Linear(hidden_sizes[1], output_size),
					nn.LogSoftmax(dim=1))

# Define Clients and Server
A = Client('A', train_sets[0], val_set, deepcopy(bottom_model))
B = Client('B', train_sets[1], val_set, deepcopy(bottom_model))
C = Client('C', train_sets[2], val_set, deepcopy(bottom_model))
D = Client('D', train_sets[3], val_set, deepcopy(bottom_model))
E = Client('E', train_sets[4], val_set, deepcopy(bottom_model))
F = Client('F', train_sets[5], val_set, deepcopy(bottom_model))
server = Server(['A', 'B', 'C', 'D', 'E', 'F'], top_model)
# server = Server(['A'], top_model)

# Train the whole model
iterations = 15
for i in range(iterations):
	# Init setting before an iteration
	server.init_setting()
	print("\nIteration {} start".format(i))

	# ForwardProp and BackProp by different Clients and Server
	A.train(server, epochs=1)
	B.train(server, epochs=1)
	C.train(server, epochs=1)
	D.train(server, epochs=1)
	E.train(server, epochs=1)
	F.train(server, epochs=1)

	# Aggression for top_model and bottom_model
	server.top_aggression()
	bottom_aggression([A, B, C, D, E, F])
	# bottom_aggression([A])

# Validate the accuracy
A.val(server.top_model)