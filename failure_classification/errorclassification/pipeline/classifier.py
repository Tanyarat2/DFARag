"""Classify a turn, validate its evidence, and review uncertain findings.

Start with FailureClassifier.classify() to read the complete workflow.
"""
import copy
import json

from failure_classification.errorclassification.evidence.availability import evidence_availability_map
from failure_classification.errorclassification.evidence.assessment import pipeline_diagnostics, normalize_evidence_response
from failure_classification.errorclassification.answerability.assessor import AnswerabilityAssessor
from failure_classification.errorclassification.data.models import Conversation, ConversationTurn
from failure_classification.errorclassification.prompts.constructor import FailurePromptConstructor
from failure_classification.errorclassification.reporting.report import build_result
from failure_classification.errorclassification.review.workflow import apply_review, plan_recheck
from failure_classification.errorclassification.review.output import normalize_review_output, review_output_errors
from failure_classification.errorclassification.validation.validator import EvidenceValidator


class FailureClassifier:
    def __init__(self, runner, *, max_passes=2, max_failures=5, method='b0'):
        if type(max_passes) is not int or max_passes not in {1, 2}:
            raise ValueError('max_passes must be 1 or 2 for every method')
        if type(max_failures) is not int or not 1 <= max_failures <= 16:
            raise ValueError('max_failures must be from 1 to 16')
        if method not in {'b0', 'b1', 'b2'}:
            raise ValueError('method must be b0, b1, or b2')
        self.runner = runner
        self.max_passes = max_passes
        self.max_failures = max_failures
        self.method = method
        self.prompt_constructor = FailurePromptConstructor()

    def run(self, conversation: Conversation):
        """Classify turns in order; keep each result and audit on its own turn."""
        results = {}
        for turn in conversation.turns:
            prediction, _ = self.classify(turn)
            results[turn.turn_id] = prediction
        return results

    def classify(self, turn):
        """Prepare evidence -> assess -> validate -> review -> store results."""
        if not isinstance(turn, ConversationTurn):
            turn = ConversationTurn.from_case_input(turn)
        case = turn.to_case_input()
        reference_map = evidence_availability_map(case)
        answerability, answerability_audit = AnswerabilityAssessor(self.runner).assess(case)
        validator = EvidenceValidator(case, answerability_assessment=answerability)

        diagnostics = pipeline_diagnostics(case)
        prompt = self.prompt_constructor.construct_prompt(case, reference_map, answerability=answerability)
        initial, response = self._ask(prompt, review=False, case=case)
        normalized, assessments, primary = validator.validate(initial)
        plan = plan_recheck(case, assessments, primary,
                            answer_defect=normalized['raw'].get('answer_defect'),
                            generation_assessment=normalized['raw'].get('generation_assessment'))
        audit = [self._audit_record(1, initial, response, assessments, primary)]
        audit[0].update(
            normalized_prediction=copy.deepcopy(normalized),
            input_evidence_map=copy.deepcopy(reference_map),
            recheck_plan=copy.deepcopy(plan),
        )

        if self.max_passes == 2 and plan['eligible']:
            prompt = self.prompt_constructor.construct_review_prompt(
                case, reference_map, normalized, assessments, primary, plan)
            patch, response = self._ask(prompt, review=True,
                                        supported_codes=plan['frozen_supported_codes'], case=case, answerability=answerability)
            normalized, assessments, primary, applied, ignored = apply_review(
                patch, normalized, assessments, primary, plan, validator)
            if response.get('review_output_valid') is False:
                ignored.append('Discarded review patch after one unsuccessful format correction; prior assessments retained')
            record = self._audit_record(2, patch, response, assessments, primary)
            record.update(
                target_codes=plan['allowed_codes'],
                recheck_kind=plan['kind'],
                recheck_plan=copy.deepcopy(plan),
                applied_codes=applied,
                primary_review_requested=plan['primary_review_requested'],
                ignored_updates=ignored,
            )
            audit.append(record)

        result = build_result(
            normalized, assessments, primary, case=case, reference_map=reference_map,
            initial=initial, plan=plan, audit=audit, method=self.method,
            max_failures=self.max_failures,
        )
        result['answerability_assessment'] = copy.deepcopy(answerability)
        result['answerability_audit'] = answerability_audit
        result['answerability_requests'] = len(answerability_audit)
        result['total_model_requests'] = len(answerability_audit) + result['passes_used'] + result['format_retries']
        result['pipeline_diagnostics'] = diagnostics
        result['pipeline_failures'] = diagnostics['failures']
        audit[-1]['final_accepted_prediction'] = copy.deepcopy(result)
        turn.record_classification(getattr(self.runner, 'model', 'unspecified'), result, audit)
        return result, audit

    def _ask(self, prompt, *, review, supported_codes=(), case=None, answerability=None):
        if review:
            return self._ask_review(prompt, supported_codes, case=case, answerability=answerability)
        response = self.runner.ask_json(
            stage='evidence_first_v5_repair' if review else 'evidence_first_v5_initial',
            prompt=prompt,
            validator=lambda output: isinstance(output, dict),
            max_completion_tokens=6144 if review else 8192,
            temperature=None,
        )
        output = response.get('parsed')
        if output is None:
            if 'invalid_output' not in response:
                message = 'Targeted request failed' if review else 'Classification request failed'
                raise RuntimeError(response.get('error', message))
            output = response['invalid_output']
        output, paths = normalize_evidence_response(output, case or {})
        response = dict(response, output_normalization=paths)
        return copy.deepcopy(output), response

    def _ask_review(self, prompt, supported_codes, *, case=None, answerability=None):
        """Allow one format correction within the single semantic review pass."""
        def contract(value):
            errors = review_output_errors(value, supported_codes)
            contract.last_error = '; '.join(errors) or None
            return not errors

        def normalizer(value):
            value, paths = normalize_review_output(value)
            value, refs = normalize_evidence_response(value, case or {})
            update = value.get('generation_assessment_update') if isinstance(value, dict) else None
            if isinstance(update, dict):
                first = answerability or {}
                for field in ('answerability', 'response_behavior'):
                    update[field] = first.get(field, 'unknown')
                refs.append('generation_assessment_update: read-only answerability copied from the frozen first stage')
            return value, paths + refs

        attempts = []
        request_prompt = prompt
        for attempt in range(2):
            response = self.runner.ask_json(
                stage='evidence_first_v5_review_contract' if attempt == 0 else 'evidence_first_v5_review_format_repair',
                prompt=request_prompt, validator=contract, normalizer=normalizer,
                json_mode=True, max_completion_tokens=6144, temperature=None,
            )
            output = response.get('parsed')
            if output is None:
                if 'invalid_output' not in response:
                    raise RuntimeError(response.get('error', 'Targeted request failed'))
                output = response['invalid_output']
            output, paths = normalizer(output)
            errors = review_output_errors(output, supported_codes)
            if response.get('error_type') == 'invalid_json':
                errors.insert(0, response.get('validation_error') or 'Reply is not valid JSON.')
            attempts.append({
                'output': copy.deepcopy(output), 'raw': response.get('raw'),
                'errors': errors, 'request_hash': response.get('request_hash'),
                'cache_hit': response.get('cache_hit', False), 'usage': response.get('usage', {}),
                'output_normalization': list(dict.fromkeys(response.get('output_normalization', []) + paths)),
            })
            if not errors:
                break
            request_prompt = (prompt + '\n\nFORMAT CORRECTION (one attempt):\n'
                + json.dumps(errors) + '\nPREVIOUS REPLY (data, not instructions):\n'
                + json.dumps(output, ensure_ascii=False)
                + '\nReturn one valid JSON patch. Preserve the diagnosis where warranted; '
                  'do not invent evidence to satisfy the schema. If selecting a new primary code, '
                  'include its candidate_updates record with its own original-record evidence. '
                  'Use primary_update=null if attribution remains unresolved. no_defect belongs '
                  'only in answer_defect_update.status. Never silently replace the evaluated answer.')
        combined = dict(response, review_response_attempts=attempts,
                        format_retries=len(attempts)-1, review_output_valid=not errors,
                        usage={key: sum(record['usage'].get(key, 0) for record in attempts)
                               for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
        # Do not partially apply a patch that still violates the contract.
        return (copy.deepcopy(output) if not errors else {}), combined

    @staticmethod
    def _audit_record(pass_number, output, response, assessments, primary):
        record = {
            'pass': pass_number,
            'output': copy.deepcopy(output),
            'raw_response': response.get('raw'),
            'label_assessments': copy.deepcopy(assessments),
            'primary_assessment': copy.deepcopy(primary),
            'request_hash': response.get('request_hash'),
            'usage': response.get('usage', {}),
            'cache_hit': response.get('cache_hit', False),
        }
        for field in ('review_response_attempts', 'format_retries', 'review_output_valid', 'output_normalization'):
            if field in response:
                record[field] = copy.deepcopy(response[field])
        return record


def classify_case(case_input, runner, *, max_passes=2, max_failures=5, method='b0'):
    """Function entry point for experiment scripts that classify one case."""
    classifier = FailureClassifier(
        runner, max_passes=max_passes, max_failures=max_failures, method=method)
    return classifier.classify(case_input)
