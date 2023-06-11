import torch.optim as optim
from model import Net
from data import trainloader, testloader
import torch
import torch.nn as nn

def train(net):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    for epoch in range(10):  # Change the number of epochs if needed
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished training')

def test(net):
    # Test the network on the test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and log the accuracy metric
    accuracy = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %.2f %%' % accuracy)

if __name__ == '__main__':
    net = Net()  # Create an instance of the neural network
    train(net)  # Train the network
    test(net)   # Test the network