import pandas as pd

from src.constants.dataset import Features, Names

from tqdm import tqdm


class FeatureProcessor:
    TARGET_COLUMN = 'evadiu'

    @staticmethod
    def process(dataset: pd.DataFrame):
        available_features = set()
        total = len(Features.ALL)

        print("\n🔍 Verificando presença das features no dataset...\n")

        # usa tqdm para exibir progresso visual
        for feature in tqdm(Features.ALL, desc="Verificando features", ncols=80, colour="cyan"):
            if feature in dataset.columns:
                tqdm.write(f"✅ {feature}")
                available_features.add(feature)
            else:
                tqdm.write(f"❌ {feature} — não encontrada")

        print("\n" + "—" * 60)
        found = len(available_features)

        print(f"📊 Resultado: {found}/{total} features encontradas ({found / total:.0%})")
        if found == total:
            print("🎉 Todas as features necessárias estão disponíveis!\n")
        elif found > 0:
            print("⚠️ Algumas features estão faltando — o modelo pode ter performance reduzida.\n")

            missing_features = set(Features.ALL) - available_features
            for feature in missing_features:
                print(f" - {feature}")
        else:
            print("🚨 Nenhuma feature esperada foi encontrada! Verifique o pré-processamento.\n")

        comparable_pattern = 'Você já fez alguma evasão (transferência) de curso?_bin'
        robust_features_filter = filter(lambda x: comparable_pattern not in x, available_features)
        robust_features = list(robust_features_filter)

        X_features = [f for f in robust_features if f in dataset.columns]

        X = dataset[X_features]
        y = dataset[FeatureProcessor.TARGET_COLUMN]

        return X, y, robust_features


class NamesProcessor:
    @staticmethod
    def process(robust_features: list):
        return [Names.LEGIBLE_NAMES.get(f, f) for f in robust_features]
