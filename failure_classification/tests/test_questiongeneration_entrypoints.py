"""Restored question-generation entry points accept external project assets."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from failure_classification.questiongeneration import xie_method, dynamic_followup_q


class QuestionGenerationEntrypointTests(unittest.TestCase):
    def test_b1_forwards_dataset_output_model_and_workbook_paths(self):
        args=['xie','--input','cases.json','--output-dir','out','--limit','3','--model','chosen',
              '--type-workbook','type.xlsx','--coverage-workbook','coverage.xlsx']
        with patch('sys.argv',args), patch.object(xie_method,'run_xie_method',return_value={}) as run,\
             contextlib.redirect_stdout(io.StringIO()):
            xie_method.main()
        self.assertEqual(run.call_args.kwargs, dict(input_path=Path('cases.json'),output_dir=Path('out'),
            limit=3,model='chosen',type_workbook=Path('type.xlsx'),coverage_workbook=Path('coverage.xlsx')))

    def test_b2_forwards_external_paths_and_custom_rag_adapter_without_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'input.json';source.write_text(json.dumps([{'query_id':'one','question':'Q?'}]))
            ask=lambda question:'Offline RAG response'
            diagnostic=SimpleNamespace(generate_text=lambda *args:'',review_question=lambda *args:None)
            args=['dynamic','--input',str(source),'--output-dir',str(root/'out'),'--limit','1',
                  '--model','chosen','--review-model','reviewer','--rag-adapter','custom_rag:ask_rag']
            with patch('sys.argv',args),patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic-test-key'}),\
                 patch('importlib.import_module',return_value=SimpleNamespace(ask_rag=ask)),\
                 patch('openai.OpenAI'),patch.object(dynamic_followup_q,'DiagnosticModelRunner',return_value=diagnostic),\
                 patch.object(dynamic_followup_q,'run_dynamic_method') as run:
                dynamic_followup_q.main()
            self.assertEqual(run.call_args.kwargs['output_dir'],root/'out')
            self.assertEqual(run.call_args.kwargs['generation_model'],'chosen')
            self.assertEqual(run.call_args.kwargs['reviewer_model'],'reviewer')
            self.assertIs(run.call_args.args[2],ask)
