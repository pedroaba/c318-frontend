import shap

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

from src import setup
from src.loaders.dataset import DatasetLoader
from src.loaders.model import ModelLoader
from src.constants.dataset import Dataset, Names
from src.process.feature import FeatureProcessor, NamesProcessor

MODEL_PATH = "./models/best_model.pkl"


def main():
    setup()

    model_loader = ModelLoader(MODEL_PATH)
    model = model_loader.load()

    dataset_loader = DatasetLoader()
    encoded_form_df = dataset_loader.load(Dataset.ENCODED_FORM)

    print(model)
    print(encoded_form_df.head())

    X, y, robust_feature = FeatureProcessor.process(encoded_form_df)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Robust features: {robust_feature}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    print(f"Dados preparados:")
    print(f" - Treino: {X_train.shape}")
    print(f" - Teste: {X_test.shape}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_values_plot = shap_values[:, :, 1]

    robust_legible_names = NamesProcessor.process(robust_feature)
    print(f"Robust legible names: {robust_legible_names}")

    print("1. Creating a global importance chart...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_plot, X_test,
                      feature_names=robust_legible_names,
                      plot_type="bar", show=False)
    plt.title("Importância Global das Features - Modelo Robusto (SHAP)", fontsize=14, fontweight='bold')
    plt.xlabel("Valor Médio Absoluto do SHAP", fontsize=12)
    plt.tight_layout()
    plt.show()

    print("2. Creating a detailed summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_plot, X_test,
                      feature_names=robust_legible_names,
                      show=False)
    plt.title("Distribuição dos Valores SHAP", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("3. Category-based feature analysis in a specific sample...")

    for sample_idx in range(X_test.shape[0]):
        shap.force_plot(
            base_value=explainer.expected_value[1],
            shap_values=shap_values_plot[sample_idx],
            features=X_test.iloc[sample_idx],
            feature_names=[Names.LEGIBLE_NAMES.get(col, col) for col in X_test.columns],
            matplotlib=True,
            figsize=(28, 4)
        )

        plt.tight_layout()
        plt.show()

    print("4. Analysis by feature category...")

    categories = {
        'Socioeconômicas': ['Situação de Moradia', 'Trabalho Atual', 'Bolsa de Estudos'],
        'Geográficas': ['Cidade de Origem', 'Frequência de Retorno', 'Natural de SRS'],
        'Acadêmicas': ['Dependências', 'Período Atual', 'Tipo de Escola'],
        'Comportamentais': ['Horas de Estudo', 'Abandono por Trabalho', 'Atividades Extracurriculares',
                            'Trancamento Anterior', 'Evasão Anterior'],
        'Demográficas': ['Faixa de Idade', 'Gênero']
    }

    importance_by_category = {}
    for category, features_cat in categories.items():
        indexes = [i for i, name in enumerate(robust_legible_names) if name in features_cat]
        if indexes:
            mean_importance = np.mean(
                [
                    np.mean(
                        np.abs(
                            shap_values_plot[:, i]
                        )
                    ) for i in indexes
                ]
            )
            importance_by_category[category] = mean_importance

    plt.figure(figsize=(10, 6))
    categories_ord = sorted(importance_by_category.items(), key=lambda x: x[1], reverse=True)
    names_cat, values_cat = zip(*categories_ord)

    bars = plt.bar(names_cat, values_cat, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731', '#5f27cd'])
    plt.title("Importância Média por Categoria de Features", fontsize=14, fontweight='bold')
    plt.ylabel("Importância SHAP Média", fontsize=12)
    plt.xticks(rotation=45, ha='right')

    for bar, value in zip(bars, values_cat):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"\n4. Ranking of importance by category:")
    for i, (category, importance) in enumerate(categories_ord, 1):
        print(f"{i}. {category}: {importance:.3f}")


if __name__ == "__main__":
    main()
