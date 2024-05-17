import torch.optim as optim
import numpy as np
from utils.data_load import Data_Pre
from torch import nn
import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset,random_split
from matplotlib import pyplot as plt
import cv2
np.set_printoptions(threshold=np.inf, linewidth=np.inf) 

#gpu설정
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else 'cpu')
device='cuda'

print("다음 기기로 학습합니다:", device)
random.seed(777)
torch.manual_seed(777)
if device=='cuda':
    torch.cuda.manual_seed_all(777)
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

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
        self.fc1 = nn.Linear(256, 128)    # Final output layer
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer5 = torch.nn.Sequential(
            self.fc1,
            nn.ReLU(),
            # torch.nn.Dropout(p=1 - self.keep_prob)
            )
        self.reg = nn.Linear(128, 1)  # Adjusted for new conv layers
        torch.nn.init.xavier_uniform_(self.reg.weight)
  


    def forward(self, x):
        out = self.layer1(x)
        out=  self.layer2(out)
        out=  self.layer3(out)
        out=  self.layer4(out)
        out = self.global_avg_pool(out)
        out = out.reshape(out.size(0), -1)   # Flatten them for FC
        out=  self.layer5(out)
        
        out = self.reg(out)
        return out

    
    
    def predict(self, x):
        return self.model.predict(x)


class CustomDataset(Dataset): 
  def __init__(self,x_data,y_data):
    self.x = x_data
    self.y = y_data

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x)

# 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self,idx):
    x=torch.from_numpy(self.x[idx]).to(device).float()
    y=torch.from_numpy(self.y[idx]).to(device).float()
    # y=y.squeeze()
    return x, y


def saveModel(net): 
    path = "/home/aims/2024/weights/heatmap/reg/final.pth" 
    torch.save(net.state_dict(), path)  
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))

    # Use numpy's advanced indexing to set elements to 1
    one_hot[np.arange(labels.size), labels.reshape(-1)] = 1

    return one_hot

# Example usage with progress printout every 10 epochs:
if __name__ == "__main__":
    # 데이터 로드
    input_arr, input_label = Data_Pre.data_load()
    # # img = X_.cpu().detach().numpy()
    # cv2.imshow("img",input_arr[0,:,:,0:3])
    # cv2.waitKey()
    dataset =CustomDataset(input_arr,input_label)
    # 데이터 로드
    total_size =input_arr.shape[0]
    train_size = int(total_size*0.8)
    valid_size = total_size - train_size
    print("total size:",train_size+valid_size)
    print("trainsets size:",train_size)
    print("validsets size:",valid_size)

    train_set, valid_set=random_split(dataset,[train_size,valid_size])

    train_loader= DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader= DataLoader(valid_set, batch_size=1, shuffle=True)

    # Initialize the CustomLSTM model
    CNN_model = CNN().to(device)

    criterion1= torch.nn.MSELoss().to(device)   
    criterion2= torch.nn.MSELoss().to(device)   

    optimizer = optim.Adam(CNN_model.parameters(), lr=0.0005)
    training_epochs =300 # Total number of epochs
    trainingEpoch_loss = []
    valid_Epoch_loss = []

    check_path1= "/home/aims/2024/weights/heatmap/reg" 
    max_val_acc = 0
    softmax = nn.Softmax(dim=None)
    # Training Mode
    CNN_model.train()

    for epoch in range(training_epochs):
        train_loss = 0.0
        print(f"------------epoch:{epoch+1}-------------")

        for X_, Y_ in train_loader: 
            optimizer.zero_grad()
            X_ = X_.permute(0, 3, 1, 2)
            outputs = CNN_model(X_)
            new_shape = len(X_)
            y_tensor_1 = Y_[:,:1].view(new_shape)
            y_tensor_2 = Y_[:,1:2].view(new_shape)

            loss1 = criterion1(outputs[:,0], y_tensor_1)
            loss2 = criterion2(outputs[:,0], y_tensor_2)
            loss = loss1+loss2
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            
            
        train_loss /= len(train_loader)

        print('train predicted:',outputs,"train answered:",Y_)

        print (f'Epoch [{epoch+1}/{training_epochs}], Train Loss: {train_loss:.4f}')
        with torch.no_grad():
            val_loss = 0.0
            total= len(val_loader)
            correct = 0
            for X_, Y_ in val_loader:  
                X_ = X_.permute(0, 3, 1, 2)
                output1 = CNN_model(X_)
                new_shape = len(X_)

                # soft_output = softmax(output1)
                # soft_output = soft_output[0,:]
                # matrix_44 = soft_output.view(4, 4).cpu()
                # matrix_44 = (matrix_44 * 255).int().numpy()
                # cell_size = 50
                # img_size = 4 * cell_size
                # visualization = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                y_tensor_1 = Y_[:,:1].view(new_shape)
                y_tensor_2 = Y_[:,1:2].view(new_shape)

                loss1 = criterion1(output1[:,0], y_tensor_1)
                loss2 = criterion2(output1[:,0], y_tensor_2)
                loss = loss1+loss2
                val_loss += loss.item()

                # print('valid predicted:',predicted,"valid answered:",answer)
            # text = " predicted:"+f"{predicted[0]}"+" answered:"+f"{answer[0]}"
            # for i in range(4):
            #     for j in range(4):
            #         # 색상 결정: 확률 값에 따라 grayscale로 적용
            #         color = matrix_44[i, j].item()
            #         cv2.rectangle(visualization, (j * cell_size, i * cell_size),
            #                     ((j + 1) * cell_size - 1, (i + 1) * cell_size - 1),
            #                     (color, color, color), -1)
            
            # cv2.putText(visualization,text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # cv2.imshow('Probability Visualization', visualization)
            # cv2.waitKey(50)
            val_loss /= len(val_loader)
            print("validation loss: {}".format(val_loss))
        if (epoch + 1) % 20 == 0:
            check_path2 = f"/{epoch + 1}.pth"
            torch.save(CNN_model.state_dict(), check_path1 + check_path2)
            print("save : " ,check_path2)
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        trainingEpoch_loss.append(train_loss)
        valid_Epoch_loss.append(val_loss)
    cv2.destroyAllWindows()
    plt.plot(trainingEpoch_loss, color='r', label='train')
    plt.plot(valid_Epoch_loss, color='b',label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train_loss','valid_loss'], loc='upper right')

    plt.show()
    print(max_val_acc)
