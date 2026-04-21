# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="838" height="569" alt="image" src="https://github.com/user-attachments/assets/5dc23b4d-6d60-4771-bb6a-96aa8acc73c5" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Prajan.ss

### Register Number: 212224230201

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)




def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      Loss=criterion(ai_brain(X_train),y_train)
      Loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(Loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {Loss.item():.6f}')



```

### Dataset Information
<img width="162" height="194" alt="image" src="https://github.com/user-attachments/assets/a30dc82f-8bc2-4ae2-bb4b-05ff8d9570ef" />


### OUTPUT
<img width="367" height="184" alt="image" src="https://github.com/user-attachments/assets/6c97b4e8-9535-4006-95ce-52fd279060c3" />

### Training Loss Vs Iteration Plot
<img width="913" height="732" alt="image" src="https://github.com/user-attachments/assets/0e541c3f-419c-411c-be52-d4705f27e0f5" />


### New Sample Data Prediction
Prediction: 9.0

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
