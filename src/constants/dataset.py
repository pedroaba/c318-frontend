from pathlib import Path


class Dataset:
    CLEAN_FORMS = Path("./database/clean_forms.csv")
    ENCODED_FORM = Path("./database/encoded_forms.csv")
    FORMS = Path("./database/forms.csv")
    TRAIN_AUGMENTED = Path("./database/train_augmented.csv")

    QUALITY_REPORT_JSON = Path("./database/ctgan_outputs/quality_report.json")


class Features:
    SOCIO_ECONOMIC = [
        "Situação de moradia_enc",
        "Você está trabalhando atualmente?_enc",
        "Você recebe bolsa ou auxílio financeiro? (Bolsa de estudos)_bin"
    ]

    GEOGRAPHIC = [
        "cidade_cod",
        "frequencia_volta_cod",
        "Sua cidade natal é Santa Rita do Sapucaí?_bin"
    ]

    ACADEMICS = [
        "dependencias_ordinal",
        "periodo_atual_enc",
        "tipo_escola_enc"
    ]

    BEHAVIORAL = [
        "horas_estudo",
        "abandono_oportunidade",
        "Participação em atividades extracurriculares_enc",
        "Você já realizou o trancamento de alguma disciplina por motivo de nota ou falta?_enc",
        "Você já fez alguma evasão (transferência) de curso?_bin"
    ]

    DEMOGRAPHICS = [
        "faixa_idade_enc",
        "genero_enc"
    ]

    ALL = [*SOCIO_ECONOMIC, *GEOGRAPHIC, *ACADEMICS, *BEHAVIORAL, *DEMOGRAPHICS]


class Names:
    LEGIBLE_NAMES = {
        "Situação de moradia_enc": "Situação de Moradia",
        "Você está trabalhando atualmente?_enc": "Trabalho Atual",
        "Você recebe bolsa ou auxílio financeiro? (Bolsa de estudos)_bin": "Bolsa de Estudos",
        "cidade_cod": "Cidade de Origem",
        "frequencia_volta_cod": "Frequência de Retorno",
        "Sua cidade natal é Santa Rita do Sapucaí?_bin": "Natural de SRS",
        "dependencias_ordinal": "Dependências",
        "periodo_atual_enc": "Período Atual",
        "tipo_escola_enc": "Tipo de Escola",
        "horas_estudo": "Horas de Estudo",
        "abandono_oportunidade": "Abandono por Trabalho",
        "Participação em atividades extracurriculares_enc": "Atividades Extracurriculares",
        "Você já realizou o trancamento de alguma disciplina por motivo de nota ou falta?_enc": "Trancamento Anterior",
        "faixa_idade_enc": "Faixa de Idade",
        "genero_enc": "Gênero",
    }
