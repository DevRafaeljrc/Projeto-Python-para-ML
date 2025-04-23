import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

# Definir a arquitetura da rede neural

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Camada de entrada para camada oculta
        self.fc2 = nn.Linear(128, 64)       # Camada oculta para camada oculta
        self.fc3 = nn.Linear(64, 10)        # Camada oculta para camada de saída

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Achatar a entrada
        x = F.relu(self.fc1(x))  # Aplicar a função de ativação ReLU
        x = F.relu(self.fc2(x))  # Aplicar a função de ativação ReLU
        x = self.fc3(x)          # Camada de saída
        return x
    
# Carregar o conjunto de dados MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalizar os dados
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Inicializar a rede neural, função de perda e otimizador
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()  # Função de perda para classificação multiclasse
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Otimizador Adam

# Treinar a rede neural
num_epochs = 5
start_time = time()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Zerar os gradientes
        output = model(data)   # Passagem para frente
        loss = criterion(output, target)  # Calcular a perda
        loss.backward()        # Passagem para trás
        optimizer.step()       # Atualizar os pesos

        if batch_idx % 100 == 0:
            print(f'Época: {epoch}, Lote: {batch_idx}, Perda: {loss.item()}')

# Salvar o modelo treinado
torch.save(model.state_dict(), 'mnist_model.pth')


# Avaliar o modelo no conjunto de teste
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)  # Obter o índice da probabilidade máxima
        total += target.size(0)  # Número total de amostras
        correct += (predicted == target).sum().item()  # Contar as previsões corretas


print(f'Acurácia do modelo no conjunto de teste: {100 * correct / total:.2f}%')


# Visualizar algumas previsões
def visualize_predictions(model, test_loader, num_images=5):
    model.eval()  # Colocar o modelo em modo de avaliação
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

    # Plotar as imagens e previsões
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Pred: {predicted[i].item()}')
        plt.axis('off')
    plt.show()


# Visualizar previsões
visualize_predictions(model, test_loader, num_images=5)

# Visualizar a perda de treinamento ao longo das épocas
def plot_training_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Perda de Treinamento')
    plt.xlabel('Número do Lote')
    plt.ylabel('Perda')
    plt.title('Perda de Treinamento ao Longo do Tempo')
    plt.legend()
    plt.show()


# Armazenar a perda de treinamento para visualização
losses = []
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Zerar os gradientes
        output = model(data)   # Passagem para frente
        loss = criterion(output, target)  # Calcular a perda
        loss.backward()        # Passagem para trás
        optimizer.step()       # Atualizar os pesos

        losses.append(loss.item())  # Armazenar a perda


# Plotar a perda de treinamento
plot_training_loss(losses)

# Visualizar os pesos do modelo
def visualize_weights(model):
    weights = model.fc1.weight.data.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(weights, cmap='gray', aspect='auto')
    plt.title('Pesos da Primeira Camada')
    plt.colorbar()
    plt.show()



# Visualizar os pesos da primeira camada
visualize_weights(model)

# Visualizar os gradientes do modelo

def visualize_gradients(model):
    gradients = model.fc1.weight.grad.data.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(gradients, cmap='gray', aspect='auto')
    plt.title('Gradientes dos Pesos da Primeira Camada')
    plt.colorbar()
    plt.show()


# Visualizar os gradientes dos pesos da primeira camada
visualize_gradients(model)


# Visualizar as ativações do modelo
def visualize_activations(model, data):
    activations = model.fc1(data.view(-1, 28 * 28)).data.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(activations, cmap='gray', aspect='auto')
    plt.title('Ativações da Primeira Camada')
    plt.colorbar()
    plt.show()
