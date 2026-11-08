"""API requests, validated JSON responses, caching, and usage for the classifier."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from openai import OpenAI

CACHE_PATH = Path(__file__).resolve().parents[2] / "cache/failure_classification_cache.json"


def _hash_json(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_json_response(raw):
    cleaned = raw.replace("<answer>", "").replace("</answer>", "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    return json.loads(cleaned)


class ClassifierRunner:
    def __init__(self, model, type_examples, coverage_examples, *, api_base_url=None,
                 api_key_env='OPENAI_API_KEY', cache_namespace=None,
                 token_limit_parameter='max_completion_tokens', cache_path=None):
        self.validate_connection_settings(model, api_key_env, cache_namespace, token_limit_parameter)
        self.initialize_api_connection(api_base_url, api_key_env, cache_namespace, token_limit_parameter)
        self.model = model
        self.type_examples = type_examples
        self.coverage_examples = coverage_examples
        self.token_limit_parameter = token_limit_parameter
        self.initialize_response_cache(cache_path)
        self.initialize_usage_counters()

    @staticmethod
    def validate_connection_settings(model, api_key_env, cache_namespace, token_limit_parameter):
        """Check model/credential-variable/cache options before client construction."""
        if not isinstance(model, str) or not model.strip():
            raise ValueError('Model must be a nonempty identifier supported by the selected API')
        if not isinstance(api_key_env, str) or not re.fullmatch(r'[A-Za-z_]\w*', api_key_env):
            raise ValueError('api_key_env must name an environment variable, not contain a credential')
        if token_limit_parameter not in {'max_completion_tokens', 'max_tokens'}:
            raise ValueError('token_limit_parameter must be max_completion_tokens or max_tokens')
        if cache_namespace is not None:
            validate_run_name(cache_namespace)

    @staticmethod
    def validate_api_base_url(api_base_url):
        """Reject endpoint URLs containing embedded credentials or query data."""
        parsed_api_url = urlsplit(api_base_url)
        if (parsed_api_url.scheme not in {'http', 'https'} or not parsed_api_url.hostname
                or parsed_api_url.username is not None or parsed_api_url.password is not None
                or parsed_api_url.query or parsed_api_url.fragment):
            raise ValueError('API base URL must be an HTTP(S) endpoint without credentials, query, or fragment')

    def initialize_api_connection(self, api_base_url, api_key_env, cache_namespace, token_limit_parameter):
        """Configure the API client and the connection identity used in cache keys."""
        api_credential = os.environ.get(api_key_env)
        if not api_credential:
            raise RuntimeError(f'{api_key_env} is not loaded; configure that API credential locally')
        resolved_api_url = (api_base_url if api_base_url is not None
                            else os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
        self.validate_api_base_url(resolved_api_url)
        self.client = OpenAI(api_key=api_credential, base_url=resolved_api_url, max_retries=2, timeout=120)
        effective_api_url = str(self.client.base_url).rstrip('/')
        self.connection_metadata = {
            'api_base_url': effective_api_url, 'api_key_env': api_key_env,
            'cache_namespace': cache_namespace, 'token_limit_parameter': token_limit_parameter,
        }
        uses_default_connection = (effective_api_url == 'https://api.openai.com/v1'
                                   and api_key_env == 'OPENAI_API_KEY' and cache_namespace is None
                                   and token_limit_parameter == 'max_completion_tokens')
        self._cache_identity = None if uses_default_connection else self.connection_metadata
        self._api_key_for_redaction = api_credential

    def initialize_response_cache(self, cache_path):
        """Load the response cache, treating missing or invalid JSON as empty."""
        self.cache_path = Path(cache_path) if cache_path is not None else CACHE_PATH
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.cache = json.loads(self.cache_path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}

    def initialize_usage_counters(self):
        """Start invocation request, cache-hit and token counters at zero."""
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        self.api_calls = 0
        self.cache_hits = 0

    def request_hash(self, *, stage, prompt, temperature=None, json_mode=False):
        """Identify an exact model/prompt request and optional connection scope."""
        request_data = {'stage': stage, 'model': self.model, 'temperature': temperature, 'prompt': prompt}
        if json_mode:
            request_data['response_format'] = {'type': 'json_object'}
        connection_identity = getattr(self, '_cache_identity', None)
        if connection_identity is not None:
            request_data['connection'] = connection_identity
        return _hash_json(request_data)

    def save_response_cache(self):
        """Replace the cache file atomically after writing complete JSON."""
        cache_file_path = getattr(self, 'cache_path', CACHE_PATH)
        temporary_cache_path = cache_file_path.with_suffix('.tmp')
        temporary_cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary_cache_path.replace(cache_file_path)

    def _save_cache(self):
        """Compatibility entry point for existing response-recovery code."""
        self.save_response_cache()

    def get_valid_cached_json_response(self, cache_key, validator):
        """Return a cached reply only when it passes the current validator."""
        cached_response = self.cache.get(cache_key)
        if cached_response and validator(cached_response.get('parsed')):
            self.cache_hits += 1
            return {**cached_response, 'request_hash': cache_key, 'cache_hit': True, 'error': None}
        return None

    def build_chat_completion_arguments(self, prompt, max_completion_tokens, temperature, json_mode=False):
        """Build the API arguments without making a request."""
        request_arguments = {
            'model': self.model,
            getattr(self, 'token_limit_parameter', 'max_completion_tokens'): max_completion_tokens,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ],
        }
        if temperature is not None:
            request_arguments['temperature'] = temperature
        if json_mode:
            request_arguments['response_format'] = {'type': 'json_object'}
        return request_arguments

    def record_chat_completion_usage(self, response):
        """Add one returned response's token usage to the invocation totals."""
        response_usage = response.usage
        token_usage = {
            field: int(getattr(response_usage, field, 0) or 0)
            for field in ('prompt_tokens', 'completion_tokens', 'total_tokens')
        }
        self.api_calls += 1
        for field in self.usage:
            self.usage[field] += token_usage[field]
        return token_usage

    def build_invalid_json_result(self, raw_reply, token_usage, cache_key, parsing_error):
        """Represent a reply that cannot be decoded as JSON."""
        return {
            'parsed': None, 'invalid_output': raw_reply,
            'raw': raw_reply, 'raw_excerpt': raw_reply[:2500], 'usage': token_usage,
            'request_hash': cache_key, 'cache_hit': False,
            'error': 'Reply was not valid JSON', 'error_type': 'invalid_json',
            'validation_error': f'JSON parsing failed: {parsing_error}', 'output_normalization': [],
        }

    def build_invalid_schema_result(self, parsed_reply, raw_reply, token_usage, cache_key,
                                    validator, normalization_paths):
        """Represent valid JSON rejected by the evidence/schema validator."""
        return {
            'parsed': None, 'invalid_output': parsed_reply,
            'raw': raw_reply, 'raw_excerpt': raw_reply[:2500], 'usage': token_usage,
            'request_hash': cache_key, 'cache_hit': False,
            'error': 'Reply JSON did not match the expected classifier output shape',
            'error_type': 'output_validation',
            'validation_error': getattr(validator, 'last_error', None),
            'output_normalization': normalization_paths,
        }

    def save_validated_response(self, cache_key, parsed_reply, raw_reply, token_usage, normalization_paths):
        """Cache only a validated response and return its request metadata."""
        cached_response = {'parsed': parsed_reply, 'raw': raw_reply, 'usage': token_usage,
                           'output_normalization': normalization_paths}
        self.cache[cache_key] = cached_response
        self.save_response_cache()
        return {**cached_response, 'request_hash': cache_key, 'cache_hit': False, 'error': None}

    def process_chat_completion_response(self, response, cache_key, validator, normalizer):
        """Coordinate response parsing, normalization, validation and caching."""
        raw_reply = response.choices[0].message.content or ''
        token_usage = self.record_chat_completion_usage(response)
        try:
            parsed_reply = _parse_json_response(raw_reply)
        except json.JSONDecodeError as parsing_error:
            return self.build_invalid_json_result(raw_reply, token_usage, cache_key, parsing_error)
        normalization_paths = []
        if normalizer is not None:
            parsed_reply, normalization_paths = normalizer(parsed_reply)
        if not validator(parsed_reply):
            return self.build_invalid_schema_result(parsed_reply, raw_reply, token_usage, cache_key,
                                                    validator, normalization_paths)
        return self.save_validated_response(cache_key, parsed_reply, raw_reply, token_usage, normalization_paths)

    def build_request_exception_result(self, request_error, cache_key):
        """Describe a request exception while redacting the configured credential."""
        error_message = str(request_error)
        api_credential = getattr(self, '_api_key_for_redaction', None)
        if isinstance(api_credential, str) and api_credential:
            error_message = error_message.replace(api_credential, '[REDACTED]')
        error_message = re.sub(r"sk-[^\s'\"},]+", '[REDACTED]', error_message)[:500]
        return {
            'parsed': None, 'raw_excerpt': None, 'usage': {},
            'request_hash': cache_key, 'cache_hit': False,
            'error': f'{type(request_error).__name__}: {error_message}',
            'error_type': type(request_error).__name__, 'http_status': getattr(request_error, 'status_code', None),
        }

    def ask_json(self, *, stage: str, prompt: str, validator: Callable[[Any], bool],
                 max_completion_tokens: int, temperature: float | None = None,
                 normalizer: Callable | None = None, json_mode: bool = False) -> dict[str, Any]:
        """Coordinate cached or fresh JSON requests through the validation pipeline."""
        cache_key = self.request_hash(stage=stage, prompt=prompt, temperature=temperature, json_mode=json_mode)
        cached_response = self.get_valid_cached_json_response(cache_key, validator)
        if cached_response is not None:
            return cached_response
        try:
            request_arguments = self.build_chat_completion_arguments(prompt, max_completion_tokens, temperature, json_mode=json_mode)
            response = self.client.chat.completions.create(**request_arguments)
            return self.process_chat_completion_response(response, cache_key, validator, normalizer)
        except Exception as request_error:
            return self.build_request_exception_result(request_error, cache_key)


def validate_run_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("Run name must be one path component using letters, numbers, underscores, or hyphens")
    return value


def add_api_arguments(parser):
    parser.add_argument("--api-base-url", help="OpenAI-compatible API base URL; defaults to OPENAI_BASE_URL or OpenAI")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding the selected API key")
    parser.add_argument("--model-label", "--run-name", dest="run_name",
                        help="Model folder and cache namespace, for example luna or JEV")
    parser.add_argument("--token-limit-parameter", choices=["max_completion_tokens", "max_tokens"],
                        default="max_completion_tokens", help="Token-limit keyword accepted by the selected API")
