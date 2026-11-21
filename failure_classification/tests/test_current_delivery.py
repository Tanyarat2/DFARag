"""The folder can run by itself in another project without historical tools/data."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class CurrentDeliveryTests(unittest.TestCase):
    def test_isolated_copy_runs_classification_gold_review_and_viewer_offline(self):
        package = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(package, root/'failure_classification', ignore=shutil.ignore_patterns('__pycache__'))
            code = r'''
import json,sys
from pathlib import Path
from unittest.mock import patch
from failure_classification.experiments import run_document_threshold_pilot as run
from failure_classification.experiments import review_pilot_gold_labels as review
assert Path(run.__file__).resolve().is_relative_to(Path.cwd())
row={'query_id':'portable', 'question':'What was revenue?', 'answer':'$100 million',
     'rag_answer_final':'Revenue was $100 million.', 'error_category':'E4 Missed Retrieval', 'error_stage':'Retrieval',
     'gold_documents':[{'doc_id':'a','text':'Revenue was $100 million.'}],
     'indexed_reference_chunks':[{'chunk_id':'a_0','text':'Revenue was $100 million.'}],
     'retrieved':[{'chunk_id':'a_0','text':'Revenue was $100 million.'}],
     'reranked':[{'chunk_id':'a_0','text':'Revenue was $100 million.'}]}
Path('input.jsonl').write_text(json.dumps(row)+'\n')
class OfflineRunner:
    def __init__(self,*a,**kw):
        self.model='offline';self.api_calls=self.cache_hits=0;self.usage={};self.client=type('Client',(),{'close':lambda self:None})()
    def ask_json(self,**request):
        if request['stage'].startswith('answerability_first'):
            value={'requirements':[{'requirement':'Revenue', 'answerability':'answerable',
                'reason':'The supplied record states revenue of $100 million.', 'response_behavior':'answered',
                'acknowledged_limitation':False, 'answer_assertion':'Revenue was $100 million.', 'evidence':['context_1']}]}
        else:
            value={'answer_defect':{'status':'no_defect','summary':'The response gives the supplied revenue.'},
                'attribution_status':'no_failure', 'candidate_failures':[], 'generation_assessment':{
                'reason':'The response correctly repeats the supplied revenue.', 'evidence':['answer','context_1'],
                'checks':{f'E{i}':{'status':'no_defect','reason':'No distinct defect in the supplied revenue answer.'} for i in range(10,17)}}}
        return {'parsed':value}
with patch.object(run,'ClassifierRunner',OfflineRunner),patch.object(sys,'argv',['run','--input','input.jsonl','--output','out']):run.main()
with patch.object(sys,'argv',['review','--run-dir','out','--input','input.jsonl']):review.main()
assert json.loads(Path('out/predictions_by_query.json').read_text())['portable']['e9_assessment']['status']=='no_failure'
assert json.loads(Path('out/gold_label_reviews.json').read_text())['portable']['verdict']=='questionable'
'''
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(root))
            completed = subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            viewer = subprocess.run([sys.executable, '-m',
                'failure_classification.experiments.build_document_threshold_viewer', '--run-dir','out'],
                cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(viewer.returncode, 0, viewer.stderr)
            html = (root/'out/viewer.html').read_text()
            self.assertIn('E4 Missed Retrieval', html)
            self.assertIn('questionable', html)
            self.assertFalse((root/'sites').exists())
