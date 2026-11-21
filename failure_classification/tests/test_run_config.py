"""YAML validation, overrides and propagation into actual classification runs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from failure_classification.errorclassification.configuration.config import load_settings, DEFAULT_CONFIG
from failure_classification.experiments import run_document_threshold_pilot as experiment


class RunConfigurationTests(unittest.TestCase):
    def config(self, root, **values):
        path = root/'settings.yaml'
        path.write_text(yaml.safe_dump({'error_classification': dict(
            input_file='input.jsonl', output_dir='run', **values)}))
        return path

    def test_shipped_config_and_paths(self):
        settings = load_settings(DEFAULT_CONFIG)
        self.assertEqual(settings.model_name, 'gpt-4o-mini')
        self.assertEqual(settings.max_failures, 16)
        self.assertEqual(settings.max_passes, 2)
        self.assertEqual(settings.limit, 10)
        self.assertTrue(settings.input_file.is_absolute())

    def test_config_paths_are_relative_to_file_and_overrides_to_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            path=self.config(root, max_failures=3)
            settings=load_settings(path, overrides={'output_dir': 'override-output', 'model_name': 'chosen-model', 'limit': None})
            self.assertEqual(settings.input_file, root/'input.jsonl')
            self.assertEqual(settings.output_dir, (Path.cwd()/'override-output').resolve())
            self.assertEqual(settings.model_name, 'chosen-model')
            self.assertIsNone(settings.limit)
            self.assertEqual(settings.max_failures, 3)

    def test_null_limit_and_numeric_case_ids(self):
        with tempfile.TemporaryDirectory() as d:
            settings=load_settings(self.config(Path(d), limit=None, case_ids=[2182,'2207']))
            self.assertIsNone(settings.limit)
            self.assertEqual(settings.case_ids,['2182','2207'])

    def test_invalid_and_unused_settings_are_rejected(self):
        for option in [dict(max_failures=0),dict(max_failures=17),dict(max_passes=3),dict(workers=0),
                       dict(limit=-1),dict(max_failures=True),dict(case_ids='2182'),dict(num_rounds=10),
                       dict(api_key_env='not a variable'),dict(token_limit_parameter='invented')]:
            with self.subTest(option=option), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):load_settings(self.config(Path(d),**option))

    def test_secrets_cannot_be_embedded_in_endpoint_url(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_settings(self.config(Path(d),api_base_url='https://user:password@example.com/v1'))

    def test_yaml_settings_reach_runner_classifier_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source={'case_id':'settings', 'original_question':'What was revenue?',
                'original_rag_answer':'Revenue was $100 million.',
                'generation_context':[{'chunk_id':'a_0','text':'Revenue was $100 million.'}],
                'retrieved_candidates':[{'chunk_id':'a_0','text':'Revenue was $100 million.'}],
                'source_documents':[{'doc_id':'a','text':'Revenue was $100 million.'}],
                'indexed_chunks':[{'chunk_id':'a_0','text':'Revenue was $100 million.'}]}
            (root/'input.jsonl').write_text(json.dumps(source)+'\n')
            config=self.config(root, model_name='chosen-model',max_failures=3,max_passes=1,workers=1,
                               api_key_env='TEST_API_KEY', token_limit_parameter='max_tokens')
            connections=[]
            class OfflineRunner:
                def __init__(self,model,*args,**kwargs):
                    connections.append((model,kwargs));self.model=model;self.api_calls=self.cache_hits=0;self.usage={}
                    self.client=type('Client',(),{'close':lambda self:None})()
                def ask_json(self,**request):
                    if request['stage'].startswith('answerability_first'):
                        return {'parsed':{'requirements':[{'requirement':'Revenue', 'answerability':'answerable',
                            'reason':'The provided record explicitly states revenue of $100 million.',
                            'response_behavior':'answered','acknowledged_limitation':False,
                            'answer_assertion':source['original_rag_answer'],'evidence':['context_1']}]}}
                    return {'parsed':{'answer_defect':{'status':'no_defect','summary':'The answer matches the supplied revenue.'},
                        'attribution_status':'no_failure','generation_assessment':{
                        'reason':'The answer correctly uses the stated revenue.', 'evidence':['answer','context_1'],
                        'checks':{f'E{i}':{'status':'no_defect','reason':'No distinct defect in the supplied revenue answer.'} for i in range(10,17)}}}}
            actual=experiment.FailureClassifier;classifier_options=[]
            def capture(runner,**kwargs):
                classifier_options.append(kwargs)
                return actual(runner,**kwargs)
            with patch.object(experiment,'ClassifierRunner',OfflineRunner),patch.object(experiment,'FailureClassifier',capture),\
                 patch('sys.argv',['run','--config',str(config)]),contextlib.redirect_stdout(io.StringIO()):
                experiment.main()
            self.assertEqual(connections[0][0],'chosen-model')
            self.assertEqual(connections[0][1]['api_key_env'],'TEST_API_KEY')
            self.assertEqual(connections[0][1]['token_limit_parameter'],'max_tokens')
            self.assertEqual(classifier_options,[{'max_passes':1,'max_failures':3,'method':'b0'}])
            manifest=json.loads((root/'run/manifest.json').read_text())
            self.assertEqual(manifest['max_failures'],3)
            self.assertEqual(manifest['max_passes'],1)
            self.assertEqual(manifest['model'],'chosen-model')
            self.assertEqual(manifest['input_file'],str(root/'input.jsonl'))
            resolved=json.loads((root/'run/resolved_config.json').read_text())
            self.assertEqual(resolved['error_classification']['max_failures'],3)
