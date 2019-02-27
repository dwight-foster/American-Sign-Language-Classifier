import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np

import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

#Defining the test and train loaders
train_data = datasets.ImageFolder(r'Drive/asl-alphabet-test', transform=transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor()]))
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(0.5 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16,sampler = train_sampler)
test_loader = torch.utils.data.DataLoader(train_data, batch_size=16,sampler = valid_sampler)


#Importing the model
#I used vgg16 but you can use something else
vgg16 = models.vgg16(pretrained=True)
#Change the last layer to output 29 because that is the number of labels
vgg16.classifier[6] = nn.Linear(4096,29)
vgg16 =vgg16.cuda()
print(vgg16)

#Defining optimizer and loss
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#Train Loop
epochs = 20
for e in range(epochs):
    train_loss = 0.0
    correct = 0.0
    total = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        output = vgg16(images)
        
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        print(train_loss)
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(labels.data.view_as(pred))).cpu().numpy())
        total += images.size(0)
    print(e)
    print('\nTrain Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
#Test loop coming soon
