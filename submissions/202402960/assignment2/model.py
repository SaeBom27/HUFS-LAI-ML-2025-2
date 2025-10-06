import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, num_classes=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 784 -> 100
            nn.ReLU(),                           # 활성화 함수
            nn.Linear(hidden_size, num_classes)  # 100 -> 10
        )

    def forward(self, x):
        """
        순전파 함수 // forward propagation
        x: 입력 텐서 (batch_size, 784)
        return: 출력 텐서 (batch_size, 10)
        """
        return self.layers(x)

# 데이터 전처리 관련 함수
def get_transforms(mean=0.1307, std=0.3081):
    """
    이미지 정규화를 포함한 Transform 반환
    """
    return transforms.Compose([
        transforms.ToTensor(),                    # PIL Image → Tensor (0~1)
        transforms.Normalize((mean,), (std,))     # 평균과 표준편차로 정규화
    ])

def transform_dataset(dataset, transform):
    """
    Dataset에 transform을 적용하는 함수
    """
    def transform_fn(batch):
        images = [transform(img).view(-1) for img in batch["image"]]  # flatten to 784
        return {
            "image": torch.stack(images),
            "label": torch.tensor(batch["label"])
        }
    return dataset.with_transform(transform_fn)

# 훈련 및 평가 함수
def train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, nb_epochs, device, verbose=True):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print("=== 훈련 시작 ===\n")

    for epoch in range(nb_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if verbose and (batch_idx + 1) % 100 == 0:
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100 * correct_train / total_train
                print(f"Epoch [{epoch+1}/{nb_epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")
                print(f"  Loss: {current_loss:.4f}, Train Acc: {current_acc:.2f}%")

        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)

        print(f"\nEpoch [{epoch+1}/{nb_epochs}] 훈련 완료:")
        print(f"  평균 Loss: {epoch_loss:.4f}")
        print(f"  훈련 정확도: {epoch_train_acc:.2f}%")

        # 테스트 정확도 측정
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100 * correct_test / total_test
        test_accuracies.append(test_acc)
        print(f"  테스트 정확도: {test_acc:.2f}%")
        print("-" * 60)

    print(f"\n=== 훈련 완료 ===")
    print(f"최종 훈련 정확도: {train_accuracies[-1]:.2f}%")
    print(f"최종 테스트 정확도: {test_accuracies[-1]:.2f}%")
  