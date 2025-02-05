import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar os dados
file_path = "car_evaluation/car.data"
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv(file_path, names=column_names)

# Pré-processamento (converter categorias em números)
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separar dados em treino e teste
X = df.drop("class", axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Testar diferentes valores de k no KNN
knn_accuracies = {}
best_knn_model = None
best_knn_k = None
best_knn_acc = 0

for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    knn_accuracies[k] = acc_knn

    print(f"\n🔹 Resultados KNN (k={k}):\n", classification_report(y_test, y_pred_knn, zero_division=1))
    print(f"Matriz de Confusão KNN (k={k}):\n", confusion_matrix(y_test, y_pred_knn))

    if acc_knn > best_knn_acc:
        best_knn_acc = acc_knn
        best_knn_k = k
        best_knn_model = knn

# Árvore de Decisão (max_depth=8`)
tree = DecisionTreeClassifier(max_depth=8, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)

print("\n🔹 Resultados Árvore de Decisão:\n", classification_report(y_test, y_pred_tree, zero_division=1))
print("Matriz de Confusão Árvore de Decisão:\n", confusion_matrix(y_test, y_pred_tree))

if not os.path.exists('imagens'):
    os.makedirs('imagens')

# Gráfico de acurácia dos modelos
accuracies = [knn_accuracies[3], knn_accuracies[5], knn_accuracies[7], acc_tree]
labels = ["KNN (k=3)", "KNN (k=5)", "KNN (k=7)", "Árvore de Decisão"]

plt.figure(figsize=(8, 5))
sns.barplot(x=labels, y=accuracies, hue=labels, dodge=False, legend=False, palette="viridis")
plt.ylim(0.8, 1)
plt.ylabel("Acurácia")
plt.title("Comparação de Modelos")
plt.xticks(rotation=15)
plt.savefig('imagens/comparacao_modelos.png')
plt.show()

# Matriz de Confusão (normalizada) para Árvore de Decisão
cm_tree = confusion_matrix(y_test, y_pred_tree)
cm_tree_normalized = cm_tree.astype("float") / cm_tree.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_tree_normalized, annot=True, cmap="Blues", fmt=".2f")
plt.title("Matriz de Confusão Normalizada - Árvore de Decisão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.savefig('imagens/matriz_confusao_normalizada_tree.png')
plt.show()

# Matriz de Confusão KNN (normalizada para o melhor k)
best_knn_y_pred = best_knn_model.predict(X_test)
cm_knn = confusion_matrix(y_test, best_knn_y_pred)
cm_knn_normalized = cm_knn.astype("float") / cm_knn.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_knn_normalized, annot=True, cmap="Blues", fmt=".2f")
plt.title(f"Matriz de Confusão Normalizada - KNN (k={best_knn_k})")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.savefig(f'imagens/matriz_confusao_normalizada_knn_{best_knn_k}.png')
plt.show()

print("\n🔹 Acurácias dos Modelos:")
print(f"KNN (k=3): {knn_accuracies[3]:.4f}")
print(f"KNN (k=5): {knn_accuracies[5]:.4f}")
print(f"KNN (k=7): {knn_accuracies[7]:.4f}")
print(f"Árvore de Decisão: {acc_tree:.4f}")


# Nuvem de Pontos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_knn, color='blue', label='KNN', alpha=0.6)
plt.scatter(y_test, y_pred_tree, color='red', label='Árvore de Decisão', alpha=0.6)
plt.xlabel('Classe Real')
plt.ylabel('Classe Predita')
plt.title('Comparação de Previsões: KNN vs Árvore de Decisão')
plt.legend()
plt.savefig('imagens/nuvem_pontos_comparacao.png')
plt.show()

# Gráfico de Distribuição das Previsões para KNN e Árvore de Decisão
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_knn, color='blue', kde=True, label='KNN', stat='density', bins=30)
sns.histplot(y_pred_tree, color='red', kde=True, label='Árvore de Decisão', stat='density', bins=30)
plt.title("Distribuição das Previsões: KNN vs Árvore de Decisão")
plt.xlabel("Valor Predito")
plt.ylabel("Densidade")
plt.legend()
plt.savefig('imagens/distribuicao_predicoes.png')
plt.show()

# Gráfico de Importância das Features para Árvore de Decisão
importances = tree.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 6))
plt.title("Importância das Features - Árvore de Decisão")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Importância")
plt.savefig('imagens/importancia_features_tree.png')
plt.show()

