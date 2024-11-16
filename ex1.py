# resnet18_fashion_mnist_optimized.py

# 필요한 라이브러리 불러오기
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast

def main():
    # 1. 데이터셋 로드 및 전처리
    # ---------------------------------

    # 데이터 전처리 변환 정의
    transform = transforms.Compose([
        transforms.Resize(32),                   # 이미지 크기 조정 (28x28 -> 32x32)
        transforms.ToTensor(),                   # 이미지를 텐서로 변환
        transforms.Normalize((0.5,), (0.5,))     # 정규화 (평균=0.5, 표준편차=0.5)
    ])

    # Fashion MNIST 데이터셋 다운로드 및 로드
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # 데이터 로더 정의
    batch_size = 128  # 배치 크기 증가
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. 모델 인스턴스 생성
    # ---------------------------------

    # ResNet18 모델 로드 (pretrained=False, num_classes=10)
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 흑백 이미지 입력에 맞게 수정
    model.maxpool = nn.Identity()  # Max Pooling 레이어 제거
    model.fc = nn.Linear(model.fc.in_features, 10)  # 클래스 수에 맞게 출력층 수정
    print(model)

    # 3. 모델 학습 설정
    # ---------------------------------

    # 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model = model.to(device)

    # 손실 함수 및 최적화 알고리즘 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습률 스케줄러 (선택 사항)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Mixed Precision 스케일러 초기화
    scaler = GradScaler()

    # 4. 모델 학습
    # ---------------------------------

    num_epochs = 20
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(1, num_epochs + 1):
        model.train()  # 모델을 훈련 모드로 설정
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 기울기 초기화
            optimizer.zero_grad()

            # Mixed Precision 순전파
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 역전파 및 최적화
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # 검증
        model.eval()  # 모델을 평가 모드로 설정
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(test_loader.dataset)
        val_epoch_acc = val_correct / val_total
        valid_losses.append(val_epoch_loss)
        valid_accuracies.append(val_epoch_acc)

        print(f'Epoch [{epoch}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        # 학습률 스케줄러 업데이트
        scheduler.step()

    # 5. 모델 평가 및 결과 시각화
    # ---------------------------------

    # 최종 테스트 평가
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    final_test_loss = test_loss / len(test_loader.dataset)
    final_test_acc = test_correct / test_total
    print(f'\nFinal Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_acc:.4f}')

    # 학습 과정 시각화
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, valid_losses, 'ro-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Train Accuracy')
    plt.plot(epochs, valid_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
