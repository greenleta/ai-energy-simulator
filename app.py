# app.py

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import psutil
import matplotlib.pyplot as plt
from model import MLP

# 🌟 타이틀
st.title("MNIST 딥러닝 에너지 최적화 시뮬레이터 🔋")

# 🌟 사용자 입력 받기
optimizer_name = st.selectbox("Optimizer 선택", ["SGD", "Adam", "RMSprop"])
lr = st.slider("학습률", 0.0001, 0.1, 0.01)
epochs = st.slider("학습 반복 횟수", 1, 10, 3)
batch_size = st.slider("배치 크기", 32, 256, 64)

# 🌟 데이터셋 불러오기
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 🌟 모델, 손실 함수, 옵티마이저 설정
model = MLP()
criterion = nn.CrossEntropyLoss()

if optimizer_name == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
elif optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
else:
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

# 🌟 학습 루프
losses = []
accuracies = []

start_time = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    losses.append(epoch_loss)
    accuracies.append(epoch_acc)

end_time = time.time()
training_time = end_time - start_time
cpu_energy_est = training_time * psutil.cpu_count() * 0.01  # 단위: Wh 가정

# 🌟 손실 함수 그래프 시각화
st.subheader("📉 손실 함수 변화")
fig1, ax1 = plt.subplots()
ax1.plot(range(1, epochs + 1), losses, marker='o')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss vs Epoch")
st.pyplot(fig1)

# 🌟 정확도 그래프 시각화
st.subheader("📈 정확도 변화")
fig2, ax2 = plt.subplots()
ax2.plot(range(1, epochs + 1), [a * 100 for a in accuracies], color='green', marker='o')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy vs Epoch")
st.pyplot(fig2)

# 🌟 결과 요약
st.metric("최종 정확도", f"{accuracies[-1]*100:.2f} %")
st.metric("에너지 소비 추정 (Wh)", f"{cpu_energy_est:.4f}")
