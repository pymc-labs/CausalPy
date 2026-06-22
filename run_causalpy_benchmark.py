"""Script de benchmark do CausalPy para medição de emissões com CodeCarbon.

Ponto de entrada por INSTANCIAÇÃO DIRETA: ao instanciar
``DifferenceInDifferences``, o próprio ``__init__`` já dispara toda a
computação (validação dos dados, construção das matrizes de design,
ajuste do modelo via PyMC e cálculo do impacto causal) — não é
necessário chamar nenhum método separado depois.

Use este arquivo como o valor da variável SCRIPT em
extract_metrics_before_codecarbon.py:

    SCRIPT = "./run_causalpy_benchmark.py"
"""

import causalpy as cp

if __name__ == "__main__":
    # Dataset de exemplo de Difference in Differences, incluído no pacote.
    df = cp.load_data("did")

    seed = 42

    # A instanciação abaixo já executa todo o trabalho pesado: ajuste do
    # modelo (amostragem MCMC via PyMC) e cálculo do impacto causal.
    result = cp.DifferenceInDifferences(
        df,
        formula="y ~ 1 + group*post_treatment",
        time_variable_name="t",
        group_variable_name="group",
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={
                "target_accept": 0.95,
                "random_seed": seed,
                "progressbar": False,
            }
        ),
    )

    summary = result.effect_summary()
    print(summary.text)
