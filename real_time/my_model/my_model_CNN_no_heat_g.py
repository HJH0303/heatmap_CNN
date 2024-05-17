from torch import nn
import torch
class CNN_heat(torch.nn.Module):
    def __init__(self):
        super(CNN_heat, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)) #pooling layer
            )
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)) #pooling layer
            )
                
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)) #pooling layer

            )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            )

        # Fully connected layers
        self.fc1 = nn.Linear(262144, 1024)  # Adjusted for new conv layers
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(1024, 128)  # Additional FC layer
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(128, 64)  # Additional FC layer
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(64, 16)    # Final output layer
        torch.nn.init.xavier_uniform_(self.fc4.weight)

        self.layer5 = torch.nn.Sequential(
            self.fc1,
            nn.ReLU(),
            # torch.nn.Dropout(p=1 - self.keep_prob)
            )
        self.layer6 = torch.nn.Sequential(
            self.fc2,
            nn.ReLU(),
            # torch.nn.Dropout(p=1 - self.keep_prob)
            )
        self.layer7 = torch.nn.Sequential(
            self.fc3,
            nn.ReLU(),
            # torch.nn.Dropout(p=1 - self.keep_prob)
            )

    def forward(self, x):
        out = self.layer1(x)
        out=  self.layer2(out)
        out=  self.layer3(out)
        out=  self.layer4(out)

        out = out.reshape(out.size(0), -1)   # Flatten them for FC
        
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        out = self.fc4(out)
        return out

    def predict(self, x):
        return self.model.predict(x)