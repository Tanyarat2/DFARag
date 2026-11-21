"""Regression checks for cache, validation, retries and annotation isolation."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from failure_classification.errorclassification.api.client import ClassifierRunner






def response_with_content(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5))


class ClassifierExecutionTests(unittest.TestCase):
    def configure_runner(self, directory, content):
        runner = ClassifierRunner.__new__(ClassifierRunner)
        runner.model = 'gpt-4o-mini'
        runner.cache = {}
        runner.cache_path = Path(directory) / 'responses.json'
        runner.output_root = Path(directory)
        runner._cache_identity = None
        runner._api_key_for_redaction = 'synthetic-credential'
        runner.api_calls = 0
        runner.cache_hits = 0
        runner.usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        runner.token_limit_parameter = 'max_completion_tokens'
        runner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(return_value=response_with_content(content)))))
        return runner

    def request(self, runner, validator, **options):
        return runner.ask_json(stage='regression', prompt='Original evidence', validator=validator,
                               max_completion_tokens=6000, temperature=0, **options)

    def test_valid_response_is_cached_and_repeat_does_not_call_api(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.configure_runner(directory, '{"value": "valid"}')
            validator = lambda value: value == {'value': 'valid'}
            first = self.request(runner, validator)
            second = self.request(runner, validator)
            self.assertEqual(first['parsed'], second['parsed'])
            self.assertFalse(first['cache_hit'])
            self.assertTrue(second['cache_hit'])
            self.assertEqual(runner.api_calls, 1)
            self.assertEqual(runner.cache_hits, 1)
            self.assertEqual(runner.usage['total_tokens'], 5)
            runner.client.chat.completions.create.assert_called_once()
            self.assertEqual(json.loads(runner.cache_path.read_text()), runner.cache)

    def test_invalid_json_is_not_cached_and_keeps_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.configure_runner(directory, 'This is not JSON')
            result = self.request(runner, lambda value: True)
            self.assertIsNone(result['parsed'])
            self.assertEqual(result['error_type'], 'invalid_json')
            self.assertEqual(result['invalid_output'], 'This is not JSON')
            self.assertEqual(result['usage']['total_tokens'], 5)
            self.assertFalse(runner.cache)
            self.assertEqual(runner.api_calls, 1)

    def test_schema_rejection_preserves_evidence_error_and_is_not_cached(self):
        def reject_reply(value):
            reject_reply.last_error = 'Exact evidence quote is missing.'
            return False
        with tempfile.TemporaryDirectory() as directory:
            runner = self.configure_runner(directory, '{"value": "invalid"}')
            result = self.request(runner, reject_reply)
            self.assertEqual(result['validation_error'], 'Exact evidence quote is missing.')
            self.assertEqual(result['error_type'], 'output_validation')
            self.assertEqual(result['invalid_output'], {'value': 'invalid'})
            self.assertFalse(runner.cache)

    def test_normalization_is_recorded_before_validation_and_caching(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.configure_runner(directory, '{"legacy": "valid"}')
            result = self.request(runner, lambda value: value == {'value': 'valid'},
                                  normalizer=lambda value: ({'value': value['legacy']}, ['legacy']))
            self.assertEqual(result['parsed'], {'value': 'valid'})
            self.assertEqual(result['output_normalization'], ['legacy'])
            self.assertIn('legacy', result['raw'])

    def test_transport_exception_redacts_credential_without_counting_a_response(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.configure_runner(directory, '')
            runner.client.chat.completions.create.side_effect = RuntimeError('synthetic-credential is invalid')
            result = self.request(runner, lambda value: True)
            self.assertNotIn('synthetic-credential', result['error'])
            self.assertIn('[REDACTED]', result['error'])
            self.assertEqual(result['error_type'], 'RuntimeError')
            self.assertEqual(runner.api_calls, 0)
            self.assertFalse(runner.cache)



    def test_runner_setup_loads_cache_and_preserves_default_connection_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / 'responses.json'
            cache_path.write_text(json.dumps({'existing': {'parsed': {'value': 'valid'}}}))
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-credential'}, clear=True):
                with patch('failure_classification.errorclassification.api.client.OpenAI') as client_factory:
                    client_factory.return_value.base_url = 'https://api.openai.com/v1/'
                    runner = ClassifierRunner('gpt-4o-mini', '', '', cache_path=cache_path)
                    self.assertIsNone(runner._cache_identity)
                    self.assertIn('existing', runner.cache)
                    self.assertEqual(runner.api_calls, 0)
                    client_factory.assert_called_once_with(api_key='synthetic-credential',
                        base_url='https://api.openai.com/v1', max_retries=2, timeout=120)

    def test_invalid_configuration_is_rejected_before_client_creation(self):
        with patch('failure_classification.errorclassification.api.client.OpenAI') as client_factory:
            with self.assertRaisesRegex(ValueError, 'nonempty identifier'):
                ClassifierRunner('', '', '')
            with self.assertRaisesRegex(ValueError, 'environment variable'):
                ClassifierRunner('gpt-4o-mini', '', '', api_key_env='invalid-variable!')
            client_factory.assert_not_called()

    def test_embedded_endpoint_credentials_are_rejected_before_client_creation(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-credential'}, clear=True):
            with patch('failure_classification.errorclassification.api.client.OpenAI') as client_factory:
                with self.assertRaisesRegex(ValueError, 'without credentials'):
                    ClassifierRunner('gpt-4o-mini', '', '', api_base_url='https://user:password@example.test/v1')
                client_factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
