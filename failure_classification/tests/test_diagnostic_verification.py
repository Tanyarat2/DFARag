"""Probe contradictions must reach subsequent planning and classification evidence."""
import json,tempfile,unittest
from pathlib import Path
from failure_classification.questiongeneration import dynamic_followup_q as dynamic
from failure_classification.questiongeneration.diagnostic_verification import verification_input,validate_verification

class VerificationTests(unittest.TestCase):
 def fixture(self):
  turn={'question':'What launched in May?','response':'A package launched in May.',
        'rag_trace':{'retrieved_candidates':[{'chunk_id':'x','text':'A package launched in April. A partnership formed in May.'}], 'generation_context':[]},'gold_answer':'SECRET'}
  result={'claims':[{'claim':turn['response'],'status':'contradicted','evidence':[{'chunk_id':'x','quote':'A package launched in April.'}],'explanation':'Launch month differs.'}], 'unresolved':['Which May event meets the requested offering?']}
  return turn,result
 def test_only_actual_passage_quotes_are_accepted(self):
  turn,result=self.fixture();payload=verification_input('Original question',turn)
  self.assertNotIn('SECRET',json.dumps(payload))
  self.assertEqual(validate_verification(result,payload),result)
  result['claims'][0]['evidence'][0]['quote']='A package launched in May.'
  with self.assertRaises(ValueError):validate_verification(result,payload)
 def test_retry_includes_rejected_quote_and_validation_feedback(self):
  turn,result=self.fixture();bad=json.loads(json.dumps(result))
  bad['claims'][0]['evidence'][0]['quote']='The package launched in April.'
  runner=object.__new__(dynamic.DiagnosticModelRunner);runner.reviewer_model='test'
  prompts=[]
  def request(role,model,prompt,limit):
   prompts.append(prompt)
   return json.dumps(bad if len(prompts)==1 else result)
  runner._request=request
  self.assertEqual(runner.verify_answer(verification_input('Original',turn)),result)
  self.assertIn('REJECTED OUTPUT',prompts[1]);self.assertIn('not an exact substring',prompts[1])
 def test_verification_reaches_next_generation_and_review(self):
  turn,result=self.fixture();seen=[]
  def generate(prompt):
   seen.append(prompt)
   return json.dumps({'claim_verification_questions':['What launched in May?'] if len(seen)==1 else [],'missing_information_questions':[]})
  rows=[{'query_id':'x','question':'Original question','original_rag_answer':'A package'}]
  with tempfile.TemporaryDirectory() as directory:
   records=dynamic.run_dynamic_method(rows,generate,lambda q:turn['rag_trace']|{'answer':turn['response']},output_dir=Path(directory),verify_answer=lambda p:result)
  self.assertEqual(records[0]['turns'][0]['evidence_verification'],result)
  self.assertIn('contradicted',seen[1]);self.assertIn('A package launched in April.',seen[1])
  review=dynamic.build_review_input([],rows[0]['question'],'A package',records[0]['turns'],'claim_verification_questions','Which event?')
  self.assertEqual(review['previous_diagnostic_turns'][0]['evidence_verification'],result)

if __name__=='__main__':unittest.main()
