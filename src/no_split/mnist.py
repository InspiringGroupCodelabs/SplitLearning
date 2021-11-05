import torch
from torchvision import datasets, transforms
from torch import nn, optim

# Define transformations
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# Download dataset
train_set = datasets.MNIST('../datasets/MNIST/train', download=True, train=True, transform=transform)
val_set = datasets.MNIST('../datasets/MNIST/val', download=True, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)

# Define models
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

epochs = 15
for epoch in range(epochs):
	running_loss = 0
	img = 0

	for images, labels in train_loader:
		# Flatten MNIST images into a 784 long vector
		images = images.view(images.shape[0], -1)
		
		# Clean the gradients
		optimizer.zero_grad()

		# evaluate full model in one pass. 
		output = model(images)

		# calculate loss
		loss = criterion(output, labels)

		#backprop the second model
		loss.backward()
		
		#optimize the weights
		optimizer.step()
		
		running_loss += loss.item()

		img = img+1

		print("Epoch {} {} - Training loss: {}".format(epoch, img, running_loss/len(train_loader)))
	else:
		print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(train_loader)))


correct_count, all_count = 0, 0
for images, labels in val_loader:
	for i in range(len(labels)):
		img = images[i].view(1, 784)
		with torch.no_grad():
			logps = model(img)

		ps = torch.exp(logps)
		probab = list(ps.numpy()[0])
		pred_label = probab.index(max(probab))
		true_label = labels.numpy()[i]
		if(true_label == pred_label):
			correct_count += 1
		all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))