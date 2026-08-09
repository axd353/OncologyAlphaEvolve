from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def _resolve_path(base_dir: Path, raw_value: str) -> Path:
    """Resolve one path value relative to the config file directory.

    Input:
        base_dir: Directory containing the config file.
        raw_value: Absolute or relative path string from the config.

    Output:
        Expanded absolute `Path`.
    """

    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _read_required_section(raw_config: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Read and validate one required top-level JSON section.

    Input:
        raw_config: Parsed TOML dictionary.
        section_name: Required top-level section name.

    Output:
        Section dictionary, or raises `ValueError` when missing or malformed.
    """

    if section_name not in raw_config:
        raise ValueError(f"Missing required config section [{section_name}].")
    section = raw_config[section_name]
    if not isinstance(section, dict):
        raise ValueError(f"Config section [{section_name}] must be a JSON object.")
    return section


@dataclass(frozen=True)
class DatasetPairConfig:
    name: str
    has_additional_covariates: bool
    training_pickles: tuple[Path, ...] = ()
    testing_pickles: tuple[Path, ...] = ()
    oracle_train_pickle: Path | None = None
    calibration_pickle: Path | None = None
    scoring_pickle: Path | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    main_output_dir: Path
    seed_priority_path: Path
    function_to_evolve: str
    max_cycles: int
    stop_after_no_improvement_cycles: int
    random_seed: int


@dataclass(frozen=True)
class ProgramDatabaseSettings:
    functions_per_prompt: int
    num_islands: int
    cluster_sampling_temperature_init: float
    cluster_sampling_temperature_period: int
    simplicity_bonus_max: float = 0.008


@dataclass(frozen=True)
class SamplerSettings:
    backend: str
    system_prompt_path: Path
    model: str
    candidates_per_island_per_cycle: int
    parallel_workers: int
    temperature: float | None
    max_output_tokens: int | None


@dataclass(frozen=True)
class EvaluatorSettings:
    backend: str
    metric: str
    oracle_train_fraction: float
    preprocessed_dirname: str
    calibration_penalties: tuple[float, ...]
    calibration_partitions: int
    scoring_partitions: int
    bootstrap_iterations: int
    dataset_pairs: tuple[DatasetPairConfig, ...]


@dataclass(frozen=True)
class LoggingSettings:
    level: str


@dataclass(frozen=True)
class PriorityToolsSettings:
    module_names: tuple[str, ...]


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    experiment: ExperimentConfig
    program_database: ProgramDatabaseSettings
    sampler: SamplerSettings
    evaluator: EvaluatorSettings
    logging: LoggingSettings
    priority_tools: PriorityToolsSettings


def _parse_dataset_pairs(base_dir: Path, evaluator_section: dict[str, Any]) -> tuple[DatasetPairConfig, ...]:
    """Parse evaluator dataset-pair config entries.

    Input:
        base_dir: Directory containing the config file, used for relative paths.
        evaluator_section: Parsed `evaluator` JSON object.

    Output:
        Tuple of dataset-pair configs. Missing entries produce an empty tuple so
        deterministic smoke tests can run without real pickles.
    """

    raw_pairs = evaluator_section.get("dataset_pairs", [])
    if raw_pairs is None:
        raw_pairs = []
    if not isinstance(raw_pairs, list):
        raise ValueError("evaluator.dataset_pairs must be an array of objects.")

    dataset_pairs: list[DatasetPairConfig] = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            raise ValueError("Each evaluator.dataset_pairs entry must be a JSON object.")

        training_pickles = tuple(
            _resolve_path(base_dir, raw_path) for raw_path in raw_pair.get("training_pickles", [])
        )
        testing_pickles = tuple(
            _resolve_path(base_dir, raw_path) for raw_path in raw_pair.get("testing_pickles", [])
        )
        oracle_train_pickle = (
            _resolve_path(base_dir, raw_pair["oracle_train_pickle"])
            if "oracle_train_pickle" in raw_pair
            else None
        )
        calibration_pickle = (
            _resolve_path(base_dir, raw_pair["calibration_pickle"])
            if "calibration_pickle" in raw_pair
            else None
        )
        scoring_pickle = (
            _resolve_path(base_dir, raw_pair["scoring_pickle"])
            if "scoring_pickle" in raw_pair
            else None
        )
        dataset_pairs.append(
            DatasetPairConfig(
                name=str(raw_pair["name"]),
                has_additional_covariates=bool(raw_pair.get("has_additional_covariates", False)),
                training_pickles=training_pickles,
                testing_pickles=testing_pickles,
                oracle_train_pickle=oracle_train_pickle,
                calibration_pickle=calibration_pickle,
                scoring_pickle=scoring_pickle,
            )
        )
    return tuple(dataset_pairs)


def _validate_config(config: PipelineConfig) -> None:
    """Validate cross-section config constraints.

    Input:
        config: Parsed strongly typed pipeline config.

    Output:
        Returns `None` for valid configs, otherwise raises `ValueError` with a
        human-readable explanation.
    """

    if config.program_database.functions_per_prompt < 1:
        raise ValueError("program_database.functions_per_prompt must be at least 1.")
    if config.program_database.num_islands < 2:
        raise ValueError("program_database.num_islands must be at least 2.")
    if config.program_database.simplicity_bonus_max < 0.0:
        raise ValueError("program_database.simplicity_bonus_max must be non-negative.")
    if config.experiment.max_cycles < 1:
        raise ValueError("experiment.max_cycles must be at least 1.")
    if config.experiment.stop_after_no_improvement_cycles < 1:
        raise ValueError(
            "experiment.stop_after_no_improvement_cycles must be at least 1."
        )
    if config.sampler.candidates_per_island_per_cycle < 1:
        raise ValueError("sampler.candidates_per_island_per_cycle must be at least 1.")
    if config.sampler.parallel_workers < 1:
        raise ValueError("sampler.parallel_workers must be at least 1.")
    if not 0.0 < config.evaluator.oracle_train_fraction < 1.0:
        raise ValueError("evaluator.oracle_train_fraction must be strictly between 0 and 1.")
    if config.evaluator.backend == "procedure2" and not config.evaluator.dataset_pairs:
        raise ValueError(
            "At least one evaluator.dataset_pairs entry is required for the procedure2 evaluator."
        )
    if config.evaluator.calibration_partitions < 1:
        raise ValueError("evaluator.calibration_partitions must be at least 1.")
    if config.evaluator.scoring_partitions < 1:
        raise ValueError("evaluator.scoring_partitions must be at least 1.")
    if config.evaluator.bootstrap_iterations < 1:
        raise ValueError("evaluator.bootstrap_iterations must be at least 1.")
    if not config.evaluator.calibration_penalties:
        raise ValueError("evaluator.calibration_penalties must contain at least one value.")
    for dataset_pair in config.evaluator.dataset_pairs:
        has_raw_pickles = bool(dataset_pair.training_pickles) and bool(dataset_pair.testing_pickles)
        has_prepared_pickles = all(
            path is not None
            for path in (
                dataset_pair.oracle_train_pickle,
                dataset_pair.calibration_pickle,
                dataset_pair.scoring_pickle,
            )
        )
        if not has_raw_pickles and not has_prepared_pickles:
            raise ValueError(
                "Each evaluator.dataset_pairs entry must provide either training/testing pickles "
                "or oracle_train_pickle, calibration_pickle, and scoring_pickle."
            )


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    """Load a JSON config file into typed pipeline settings.

    Input:
        config_path: Path to the user-editable JSON config file.

    Output:
        `PipelineConfig` with resolved paths and defaulted optional settings.
    """

    resolved_config_path = Path(config_path).expanduser().resolve()
    base_dir = resolved_config_path.parent
    raw_config = json.loads(resolved_config_path.read_text())
    if not isinstance(raw_config, dict):
        raise ValueError("Top-level config must be a JSON object.")

    experiment_section = _read_required_section(raw_config, "experiment")
    program_database_section = _read_required_section(raw_config, "program_database")
    sampler_section = _read_required_section(raw_config, "sampler")
    evaluator_section = _read_required_section(raw_config, "evaluator")
    logging_section = raw_config.get("logging", {})
    priority_tools_section = raw_config.get("priority_tools", {})

    config = PipelineConfig(
        config_path=resolved_config_path,
        experiment=ExperimentConfig(
            name=str(experiment_section["name"]),
            main_output_dir=_resolve_path(base_dir, experiment_section["main_output_dir"]),
            seed_priority_path=_resolve_path(base_dir, experiment_section["seed_priority_path"]),
            function_to_evolve=str(experiment_section.get("function_to_evolve", "priority")),
            max_cycles=int(experiment_section.get("max_cycles", 10)),
            stop_after_no_improvement_cycles=int(
                experiment_section.get("stop_after_no_improvement_cycles", 5)
            ),
            random_seed=int(experiment_section.get("random_seed", 0)),
        ),
        program_database=ProgramDatabaseSettings(
            functions_per_prompt=int(program_database_section.get("functions_per_prompt", 2)),
            num_islands=int(program_database_section.get("num_islands", 8)),
            cluster_sampling_temperature_init=float(
                program_database_section.get("cluster_sampling_temperature_init", 0.1)
            ),
            cluster_sampling_temperature_period=int(
                program_database_section.get("cluster_sampling_temperature_period", 30_000)
            ),
            simplicity_bonus_max=float(
                program_database_section.get("simplicity_bonus_max", 0.008)
            ),
        ),
        sampler=SamplerSettings(
            backend=str(sampler_section.get("backend", "openai")),
            system_prompt_path=_resolve_path(base_dir, sampler_section["system_prompt_path"]),
            model=str(sampler_section.get("model", "gpt-5.4-mini")),
            candidates_per_island_per_cycle=int(
                sampler_section.get("candidates_per_island_per_cycle", 4)
            ),
            parallel_workers=int(
                sampler_section.get(
                    "parallel_workers",
                    program_database_section.get("num_islands", 8),
                )
            ),
            temperature=(
                float(sampler_section["temperature"])
                if "temperature" in sampler_section
                else None
            ),
            max_output_tokens=(
                int(sampler_section["max_output_tokens"])
                if "max_output_tokens" in sampler_section
                else None
            ),
        ),
        evaluator=EvaluatorSettings(
            backend=str(evaluator_section.get("backend", "procedure2")),
            metric=str(evaluator_section.get("metric", "roc_auc")),
            oracle_train_fraction=float(evaluator_section.get("oracle_train_fraction", 0.8)),
            preprocessed_dirname=str(evaluator_section.get("preprocessed_dirname", "preprocessed")),
            calibration_penalties=tuple(
                float(value)
                for value in evaluator_section.get("calibration_penalties", [1.0])
            ),
            calibration_partitions=int(evaluator_section.get("calibration_partitions", 1)),
            scoring_partitions=int(evaluator_section.get("scoring_partitions", 1)),
            bootstrap_iterations=int(evaluator_section.get("bootstrap_iterations", 200)),
            dataset_pairs=_parse_dataset_pairs(base_dir, evaluator_section),
        ),
        logging=LoggingSettings(level=str(logging_section.get("level", "INFO"))),
        priority_tools=PriorityToolsSettings(
            module_names=tuple(str(value) for value in priority_tools_section.get("module_names", []))
        ),
    )
    _validate_config(config)
    return config
