"""Validated run settings shared by YAML configuration and CLI overrides."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / 'conf/config-classification.yaml'
PATH_FIELDS = {'input_file', 'output_dir', 'reuse_cache_from'}


@dataclass(frozen=True)
class ClassificationSettings:
    model_name: str = 'gpt-4o-mini'
    max_failures: int = 16
    max_passes: int = 2
    method: str = 'b0'
    input_file: Path | None = None
    output_dir: Path | None = None
    limit: int | None = 10
    case_ids: list[str] | None = None
    workers: int = 4
    reuse_cache_from: Path | None = None
    api_base_url: str | None = None
    api_key_env: str = 'OPENAI_API_KEY'
    token_limit_parameter: str = 'max_completion_tokens'

    def to_dict(self):
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


def _path(value, base, key):
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f'{key} must be a nonempty path or null')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_settings(config_path=DEFAULT_CONFIG, *, overrides=None):
    """YAML paths use its directory; explicit CLI paths use the working directory."""
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding='utf-8') as source:
        document = yaml.safe_load(source)
    if not isinstance(document, dict) or set(document) != {'error_classification'}:
        raise ValueError('Config must contain one error_classification mapping')
    values = document['error_classification']
    if not isinstance(values, dict):
        raise ValueError('error_classification must be a mapping')
    values = dict(values)
    allowed = {field.name for field in fields(ClassificationSettings)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError('Unknown error_classification settings: ' + ', '.join(sorted(unknown)))
    for key in PATH_FIELDS & values.keys():
        values[key] = _path(values[key], path.parent, key)
    for key, value in (overrides or {}).items():
        if key not in allowed:
            raise ValueError(f'Unknown override: {key}')
        values[key] = _path(value, Path.cwd(), key) if key in PATH_FIELDS else value
    settings = ClassificationSettings(**values)
    if not isinstance(settings.model_name, str) or not settings.model_name.strip():
        raise ValueError('model_name must be a nonempty string')
    for key, low, high in [('max_failures', 1, 16), ('max_passes', 1, 2), ('workers', 1, None), ('limit', 1, None)]:
        value = getattr(settings, key)
        if key == 'limit' and value is None:
            continue
        if type(value) is not int or value < low or (high is not None and value > high):
            raise ValueError(f'{key} must be an integer from {low}' + (f' to {high}' if high is not None else ' or greater'))
    if settings.method not in ('b0', 'b1', 'b2'):
        raise ValueError('method must be b0, b1 or b2')
    if settings.input_file is None or settings.output_dir is None:
        raise ValueError('input_file and output_dir are required')
    if settings.case_ids is not None:
        if not isinstance(settings.case_ids, list) or not all(isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip() for v in settings.case_ids):
            raise ValueError('case_ids must be a list of nonempty IDs or null')
        settings = ClassificationSettings(**dict(asdict(settings), case_ids=[str(v) for v in settings.case_ids]))
    # Validate connection options without creating a client or making a request.
    import re
    if not isinstance(settings.api_key_env, str) or not re.fullmatch(r'[A-Za-z_]\w*', settings.api_key_env):
        raise ValueError('api_key_env must name an environment variable, not contain a credential')
    if settings.token_limit_parameter not in ('max_completion_tokens', 'max_tokens'):
        raise ValueError('token_limit_parameter must be max_completion_tokens or max_tokens')
    if settings.api_base_url is not None:
        from failure_classification.errorclassification.api.client import ClassifierRunner
        if not isinstance(settings.api_base_url, str):
            raise ValueError('api_base_url must be a URL string or null')
        ClassifierRunner.validate_api_base_url(settings.api_base_url)
    return settings
