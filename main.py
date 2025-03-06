import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18의 입력 크기
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet 평균
                         [0.229, 0.224, 0.225])   # ImageNet 표준편차
])

# 데이터 경로 설정
IMAGE_DIR = '/content/AI_Trash_Train_Data'  # 로컬 디렉토리 경로로 변경

# ImageFolder를 사용하여 데이터셋 로드
full_dataset = datasets.ImageFolder(root=IMAGE_DIR, transform=transform)

# 클래스 이름 출력
print("클래스 목록:", full_dataset.classes)

# 학습용과 검증용으로 분할 (80% 학습, 20% 검증)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"전체 데이터 수: {len(full_dataset)}")
print(f"학습 데이터 수: {len(train_dataset)}")
print(f"검증 데이터 수: {len(val_dataset)}")

# 데이터로더 정의
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# ResNet18 모델 초기화
model = models.resnet18(pretrained=True)

# 마지막 완전 연결층 수정
num_features = model.fc.in_features
num_classes = len(full_dataset.classes)
model.fc = nn.Linear(num_features, num_classes)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"사용 중인 디바이스: {device}")

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 및 검증 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_acc = 0.0
    all_preds = []
    all_labels = []

    # 학습 및 검증 손실 및 정확도 기록
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

                # 혼동 행렬을 생성하기 위해 예측값과 실제 라벨을 수집
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

        # 검증 정확도가 향상되면 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet18_garbage_classifier_main3.pth')
            print(f'Best model saved with accuracy: {best_val_acc:.4f}\n')

    print('Training complete')

    # 학습 및 검증 손실과 정확도 반환
    return all_preds, all_labels, train_losses, val_losses, train_accuracies, val_accuracies

# 모델 학습
num_epochs = 20  # 원하는 에폭 수로 설정
all_preds, all_labels, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs
)

# 최고 성능 모델 로드
model.load_state_dict(torch.load('best_resnet18_garbage_classifier_main3.pth'))
model.eval()

# 혼동 행렬 및 분류 보고서 생성
conf_matrix = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=full_dataset.classes)

# 분류 보고서 출력
print("Classification Report:\n")
print(report)

# 혼동 행렬 시각화
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 정확률과 손실도의 꺽은선 그래프 출력
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(14, 5))

    # 손실도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 정확률 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 그래프 출력
plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs)

# 예측 함수 정의
def predict_image(image_path, model, transform, class_names):
    from PIL import Image  # PIL이 누락된 경우 추가

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        return class_names[preds.item()]


