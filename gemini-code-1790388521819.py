import logging
import subprocess
import sys
from pathlib import Path

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RennerPipeline")


def run_stage(stage_name: str, module_path: str) -> None:
    """
    Executa um estágio do pipeline como módulo sem alterar funções internas.
    """
    logger.info(f"=== [INÍCIO] Estágio: {stage_name} ===")
    try:
        result = subprocess.run(
            [sys.executable, "-m", module_path],
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.debug(result.stdout)
        logger.info(f"=== [CONCLUÍDO] Estágio: {stage_name} ===\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro no estágio '{stage_name}': {e.stderr}")
        raise RuntimeError(f"Falha na execução do estágio: {stage_name}") from e


def execute_pipeline():
    """
    Orquestração sequencial conforme o fluxo de arquitetura.
    """
    logger.info("Iniciando Pipeline de Dados e Inteligência Financeira/ESG")

    # 1. Ingestion
    run_stage("Ingestion", "renner_work.scripts.ingest_raw_data")

    # 2. Quality
    run_stage("Quality", "lab02_quality.clean_dataset")

    # 3. Engineering
    run_stage("Engineering", "lab01_engineering.build_dataset")

    # 4. Features
    run_stage("Features", "renner_work.scripts.build_features")

    # 5. Financial Mart
    run_stage("Financial Mart", "renner_work.scripts.build_financial_mart")

    # 6. Parallels / Convergence: Forecast & ESG -> Risk
    run_stage("Forecast Model", "renner_work.models.train_forecast")
    run_stage("ESG Processing", "renner_work.scripts.build_esg_marts")
    run_stage("Risk Modeling", "renner_work.models.risk_assessment")

    # 7. Transition Model
    run_stage("Transition Model", "renner_work.models.transition_model")

    # 8. RL Train
    run_stage("RL Training", "renner_work.models.train_rl_policy")

    # 9. RL Evaluation
    run_stage("RL Evaluation", "renner_work.models.evaluate_rl_policy")

    # 10. GIA (Governança, Impacto e Alertas)
    run_stage("GIA Alerts & Auditing", "renner_work.scripts.gia_alerts")

    # 11. Reporting
    run_stage("Reporting & Dashboards", "renner_work.scripts.generate_reports")

    logger.info("Pipeline executada com sucesso de ponta a ponta!")


if __name__ == "__main__":
    execute_pipeline()