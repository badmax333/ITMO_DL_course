import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics import F1Score, Accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_model(model, datamodule, batch_size=256, criterion=None):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    device = next(model.parameters()).device
    
    test_dataset = datamodule.test_dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    num_classes = 5
    
    f1_macro = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_weighted = F1Score(task="multiclass", num_classes=num_classes, average='weighted').to(device)
    f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average='none').to(device)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    test_loss = 0.0
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            if target.min() > 0:
                target_adjusted = target - 1
            else:
                target_adjusted = target
            
            loss = criterion(output, target_adjusted)
            test_loss += loss.item()
            
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target_adjusted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            f1_macro.update(predictions, target_adjusted)
            f1_weighted.update(predictions, target_adjusted)
            f1_per_class.update(predictions, target_adjusted)
            accuracy.update(predictions, target_adjusted)
    
    avg_test_loss = test_loss / len(test_loader)
    test_f1_macro = f1_macro.compute().cpu().item()
    test_f1_weighted = f1_weighted.compute().cpu().item()
    test_f1_per_class = f1_per_class.compute().cpu().numpy()
    test_accuracy = accuracy.compute().cpu().item()
    
    print("="*50)
    print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("="*50)
    
    print(f"\nРезультаты:")
    print("-" * 40)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test F1 Macro: {test_f1_macro:.4f}")
    print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
    print("-" * 40)
    
    print("\nF1-score по классам:")
    for class_idx in range(num_classes):
        original_rating = class_idx + 1
        print(f"  Оценка {original_rating} звезд: {test_f1_per_class[class_idx]:.4f}")
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    all_predictions_original = all_predictions + 1
    all_targets_original = all_targets + 1
    
    cm = confusion_matrix(all_targets_original, all_predictions_original, labels=[1, 2, 3, 4, 5])
    
    print("\n" + "="*50)
    print("МАТРИЦА ОШИБОК")
    print("="*50)
    
    print("\nМатрица ошибок:")
    print(" " * 8 + " ".join([f"Предск.{i}" for i in range(1, 6)]))
    for i in range(5):
        print(f"Истинн.{i+1}: " + " ".join([f"{cm[i][j]:8d}" for j in range(5)]))
    
    print("\n" + "="*50)
    print("ОТЧЕТ")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(
        all_targets_original, 
        all_predictions_original,
        target_names=['1 звезда', '2 звезды', '3 звезды', '4 звезды', '5 звезд']
    ))
    
    _plot_evaluation_results(
        all_targets_original, 
        all_predictions_original, 
        all_probabilities,
        all_predictions,
        test_f1_per_class,
        cm
    )
    
    print("\n" + "="*50)
    print("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА")
    print("="*50)
    
    print("\nPrecision и Recall по классам:")
    for class_idx in range(num_classes):
        rating = class_idx + 1
        tp = cm[class_idx, class_idx]
        total_predicted = cm[:, class_idx].sum()
        total_actual = cm[class_idx, :].sum()
        
        precision = tp / total_predicted if total_predicted > 0 else 0
        recall = tp / total_actual if total_actual > 0 else 0
        
        print(f"  Оценка {rating}: Precision={precision:.3f}, Recall={recall:.3f}")
    
    error_counts = {}
    for true, pred in zip(all_targets_original, all_predictions_original):
        if true != pred:
            error = (int(true), int(pred))
            error_counts[error] = error_counts.get(error, 0) + 1
    
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nТоп-5 самых частых ошибок:")
    print("  Истинная → Предсказанная : Количество")
    for (true, pred), count in sorted_errors:
        print(f"  {true} → {pred} : {count}")
    
    print("\n" + "="*50)
    print("ОЦЕНКА ЗАВЕРШЕНА!")
    print("="*50)


def _plot_evaluation_results(all_targets_original, all_predictions_original, 
                           all_probabilities, all_predictions,
                           test_f1_per_class, cm):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1 = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['1', '2', '3', '4', '5'],
                yticklabels=['1', '2', '3', '4', '5'],
                ax=ax1)
    ax1.set_title('Матрица ошибок')
    ax1.set_xlabel('Предсказанные оценки')
    ax1.set_ylabel('Истинные оценки')
    
    ax2 = axes[0, 1]
    unique_targets, counts_targets = np.unique(all_targets_original, return_counts=True)
    unique_preds, counts_preds = np.unique(all_predictions_original, return_counts=True)
    
    width = 0.35
    x = np.arange(len(unique_targets))
    ax2.bar(x - width/2, counts_targets, width, label='Истинные', alpha=0.7)
    ax2.bar(x + width/2, counts_preds, width, label='Предсказанные', alpha=0.7)
    ax2.set_xlabel('Оценки')
    ax2.set_ylabel('Количество')
    ax2.set_title('Распределение оценок')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(i)}' for i in unique_targets])
    ax2.legend()
    
    ax3 = axes[1, 0]
    classes = ['1★', '2★', '3★', '4★', '5★']
    f1_scores = test_f1_per_class
    bars = ax3.bar(classes, f1_scores, color='skyblue')
    ax3.set_ylim([0, 1])
    ax3.set_ylabel('F1-score')
    ax3.set_title('F1-score по классам')
    
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    ax4 = axes[1, 1]
    correct_mask = (all_predictions_original == all_targets_original)
    correct_confidences = [prob[pred] for prob, pred, correct in 
                          zip(all_probabilities, all_predictions, correct_mask) if correct]
    incorrect_confidences = [prob[pred] for prob, pred, correct in 
                            zip(all_probabilities, all_predictions, correct_mask) if not correct]
    
    ax4.hist(correct_confidences, bins=20, alpha=0.7, label='Верные', color='green')
    ax4.hist(incorrect_confidences, bins=20, alpha=0.7, label='Ошибочные', color='red')
    ax4.set_xlabel('Уверенность модели')
    ax4.set_ylabel('Количество')
    ax4.set_title('Распределение уверенности модели')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()