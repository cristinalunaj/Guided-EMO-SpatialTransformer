import torch.nn as nn
import torch.nn.functional as F

class Deep_Emotion_Baseline(nn.Module):
    def __init__(self, training=True):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        self.trainingState=training
        super(Deep_Emotion_Baseline,self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,3)
        # define dropout layer in __init__
        #self.drop_layer = nn.Dropout(p=0.5)



    def forward(self,input1):

        out = F.relu(self.conv1(input1))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out, training=self.trainingState)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out