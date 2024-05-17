import torch.optim as optim
import numpy as np
from train.utils.data_grid_load import Data_Pre
from torch import nn
import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset,random_split
from matplotlib import pyplot as plt
import cv2
from torch.optim.lr_scheduler import StepLR
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

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        path = "/home/aims/2024/weights/heatmap/gap/best.pth" 

        torch.save(model.state_dict(), path)  

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
    path = "/home/aims/2024/weights/heatmap/gap/final.pth" 
    torch.save(net.state_dict(), path)  
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))

    # Use numpy's advanced indexing to set elements to 1
    one_hot[np.arange(labels.size), labels.reshape(-1)] = 1

    return one_hot

# Example usage with progress printout every 10 epochs:
if __name__ == "__main__":
    # 데이터 로드
    input_arr, input_label2 = Data_Pre.data_load()
    input_label = one_hot_encode(input_label2,16)
    # # img = X_.cpu().detach().numpy()
    # cv2.imshow("img",input_arr[0,:,:,0:3])
    # cv2.waitKey()
    
    dataset =CustomDataset(input_arr,input_label)
    # 데이터 로드
    total_size =input_arr.shape[0]
    train_size = int(total_size*0.7)
    valid_size = total_size - train_size
    print("total size:",train_size+valid_size)
    print("trainsets size:",train_size)
    print("validsets size:",valid_size)

    train_set, valid_set=random_split(dataset,[train_size,valid_size])

    train_loader= DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader= DataLoader(valid_set, batch_size=1, shuffle=True)

    # Initialize the CustomLSTM model
    CNN_model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(CNN_model.parameters(), lr=0.0001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    training_epochs = 300 # Total number of epochs
    trainingEpoch_loss = []
    valid_Epoch_loss = []
    early_stopping = EarlyStopping(patience=5, verbose=True)
    stop_epoch = 1
    check_path1= "/home/aims/2024/weights/heatmap/gap" 
    max_val_acc = 0
    softmax = nn.Softmax(dim=None)
    # Training Mode
    if torch.cuda.device_count()>1:
        CNN_model = nn.DataParallel(CNN_model)
    CNN_model.train()
    for epoch in range(training_epochs):
        train_loss = 0.0
        print(f"------------epoch:{epoch+1}-------------")

        for X_, Y_ in train_loader: 
            optimizer.zero_grad()
            X_ = X_.permute(0, 3, 1, 2)
            outputs = CNN_model(X_)
            train_loss_in = criterion(outputs, Y_.float())
            train_loss_in.backward()
            optimizer.step()
            train_loss+=train_loss_in.item()
            _, predicted = torch.max(outputs, 1)

            answer = torch.argmax(Y_,1)

        print('train predicted:',predicted,"train answered:",answer)

        print (f'Epoch [{epoch+1}/{training_epochs}], Train Loss: {train_loss/len(train_loader):.4f}')
        with torch.no_grad():
            val_loss = 0.0
            total= len(val_loader)
            correct = 0
            for X_, Y_ in val_loader:  
                X_ = X_.permute(0, 3, 1, 2)
                output1 = CNN_model(X_)
                # soft_output = softmax(output1)
                # soft_output = soft_output[0,:]
                # matrix_44 = soft_output.view(4, 4).cpu()
                # matrix_44 = (matrix_44 * 255).int().numpy()
                # cell_size = 50
                # img_size = 4 * cell_size
                # visualization = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                _, predicted = torch.max(output1, 1)
                val_loss_in = criterion(output1, Y_).to(device)
                val_loss += val_loss_in.item()
                answer = torch.argmax(Y_,1)
                if answer[0] == predicted[0]: correct+=1
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
            max_val_acc= max(max_val_acc,correct/total*100)
            val_loss/=len(val_loader)
            print("validation accuracy:",correct/total*100)
            print("validation loss: {}".format(val_loss))
        if (epoch + 1) % 20 == 0:
            check_path2 = f"/{epoch + 1}.pth"
            torch.save(CNN_model.state_dict(), check_path1 + check_path2)
            print("save : " ,check_path2)
        train_loss /= len(train_loader)
        trainingEpoch_loss.append(train_loss)
        valid_Epoch_loss.append(val_loss)
        stop_epoch = epoch
        if early_stopping(val_loss, CNN_model):
            print("Early stopping")
            break
        # scheduler.step()

    cv2.destroyAllWindows()
    plt.plot(trainingEpoch_loss, color='r', label='train')
    plt.plot(valid_Epoch_loss, color='b',label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train_loss','valid_loss'], loc='upper right')

    plt.show()
    print("stop epoch:", stop_epoch)

    print("validation_max_Val",max_val_acc)
