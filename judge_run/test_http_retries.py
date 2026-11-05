"""Offline regression checks for provider HTTP rejections."""
import io
import unittest
import urllib.error
from unittest import mock

import judge_mtrag as judge


class HttpRetryTests(unittest.TestCase):
    def args(self):
        args = judge.parse_args([
            '--input', __file__, '--reasoning-max-tokens', '4096',
            '--max-tokens', '8192', '--adaptive-max-tokens', '32768'])
        args.api_key = 'test-secret'
        return args

    def error(self, code=429, headers=None):
        return urllib.error.HTTPError('https://example.test', code, 'Rejected',
                                      headers or {}, io.BytesIO(b'provider busy test-secret'))

    def response(self):
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'OK'}}]}

    def test_rate_limit_waits_and_preserves_budgets(self):
        with mock.patch.object(judge, '_post', side_effect=[
                self.error(headers={'Retry-After': '45'}), self.response()]) as post, \
                mock.patch.object(judge.time, 'sleep') as sleep:
            self.assertEqual(judge.ask(self.args(), 'Prompt', str, {}), 'OK')
        self.assertEqual([c.args[1]['max_tokens'] for c in post.call_args_list], [8192, 8192])
        self.assertEqual([c.args[1]['reasoning']['max_tokens'] for c in post.call_args_list],
                         [4096, 4096])
        self.assertEqual(sum(c.args[0] for c in sleep.call_args_list), 45)

    def test_final_error_explains_provider_rejection_and_redacts_key(self):
        with mock.patch.object(judge, '_post', side_effect=[self.error() for _ in range(3)]), \
                mock.patch.object(judge.time, 'sleep'):
            with self.assertRaises(RuntimeError) as caught:
                judge.ask(self.args(), 'Prompt', str, {})
        self.assertIn('HTTP 429: provider busy [REDACTED]', str(caught.exception))
        self.assertNotIn('test-secret', str(caught.exception))
        self.assertIn('last_requested_max_tokens=8192', str(caught.exception))

    def test_auth_error_stops_without_retries(self):
        with mock.patch.object(judge, '_post', side_effect=self.error(401)) as post, \
                mock.patch.object(judge.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'after 1 attempts: HTTP 401'):
                judge.ask(self.args(), 'Prompt', str, {})
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
