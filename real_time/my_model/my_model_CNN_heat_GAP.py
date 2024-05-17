from torch import nn
import torch
class CNN_heat(torch.nn.Module):
    def __init__(self):
        super(CNN_heat, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1),
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
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)  # Adjusted for new conv layers
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.classifier = nn.Linear(128, 16)
        self.layer5 = torch.nn.Sequential(
            self.fc1,
            nn.ReLU(),
            # torch.nn.Dropout(p=1 - self.keep_prob)
            )

    def forward(self, x):
        out = self.layer1(x)
        out=  self.layer2(out)
        out=  self.layer3(out)
        out=  self.layer4(out)
        out = self.global_avg_pool(out)
        out = out.reshape(out.size(0), -1)   # Flatten them for FC
        out=  self.layer5(out)
        
        out = self.classifier(out)
        return out

    def predict(self, x):
        return self.model.predict(x)