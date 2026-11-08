"""Conversation records with turn-scoped evidence and model-specific diagnoses.

Classes own the data in memory; JSON remains the on-disk format. Reference
material never becomes retrieved or generator evidence implicitly.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConversationTurn:
    """One evaluated question/answer and the evidence from that execution."""

    turn_id: str
    question: str
    answer: str
    case_id: str | None = None
    model: str | None = None
    retrieved_chunks: list[dict[str, Any]] | None = None
    generation_context: list[dict[str, Any]] | None = None
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    reference_evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    classifications: dict[str, dict[str, Any]] = field(default_factory=dict)
    _input_fields: set[str] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for name in ('turn_id', 'question', 'answer'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonempty string')
        if self.case_id is None:
            self.case_id = self.turn_id
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError('case_id must be a nonempty string')
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError('model must be a nonempty string or None')
        for name in ('retrieved_chunks', 'generation_context', 'conversation_history'):
            value = getattr(self, name)
            if value is None and name != 'conversation_history':
                continue
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                raise ValueError(f'{name} must be a list of objects' + (' or None' if name != 'conversation_history' else ''))
            setattr(self, name, copy.deepcopy(value))
        for name in ('reference_evidence', 'metadata', 'classifications'):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise ValueError(f'{name} must be an object')
            setattr(self, name, copy.deepcopy(value))
        allowed_reference_fields = {'gold_answer', 'source_documents', 'indexed_chunks'}
        if set(self.reference_evidence) - allowed_reference_fields:
            raise ValueError('reference_evidence accepts only gold_answer, source_documents, and indexed_chunks')
        reserved = {'case_id', 'original_question', 'original_rag_answer',
                    'retrieved_candidates', 'generation_context', 'conversation_history'} | allowed_reference_fields
        if set(self.metadata) & reserved:
            raise ValueError('metadata cannot override turn identity, evidence, question, or answer')

    @classmethod
    def from_case_input(cls, case: dict[str, Any]) -> ConversationTurn:
        """Adapt existing classifier inputs without changing their evidence or keys."""
        from failure_classification.errorclassification.data.case_io import without_failure_labels

        if not isinstance(case, dict):
            raise TypeError('case input must be an object')
        case = without_failure_labels(case)
        reference_keys = {'gold_answer', 'source_documents', 'indexed_chunks'}
        main_keys = {'case_id', 'original_question', 'original_rag_answer',
                     'retrieved_candidates', 'generation_context', 'conversation_history'}
        case_id = case.get('case_id')
        if case_id is None:
            raise ValueError('case_id is required')
        return cls(
            case_id=str(case_id), turn_id=str(case.get('turn_id', case_id)),
            question=case.get('original_question'), answer=case.get('original_rag_answer'),
            retrieved_chunks=case.get('retrieved_candidates'),
            generation_context=case.get('generation_context'),
            conversation_history=case.get('conversation_history', []),
            reference_evidence={key: value for key, value in case.items() if key in reference_keys},
            metadata={key: value for key, value in case.items() if key not in main_keys | reference_keys},
            _input_fields=set(case),
        )

    def to_case_input(self) -> dict[str, Any]:
        """Export original evidence only; never feed classifications back to the model."""
        from failure_classification.errorclassification.data.case_io import without_failure_labels

        case = copy.deepcopy(self.metadata)
        case.update(case_id=self.case_id, original_question=self.question, original_rag_answer=self.answer)
        optional = {'retrieved_candidates': self.retrieved_chunks,
                    'generation_context': self.generation_context,
                    'conversation_history': self.conversation_history}
        for key, value in optional.items():
            if self._input_fields is None or key in self._input_fields or value not in (None, []):
                case[key] = copy.deepcopy(value)
        case.update(copy.deepcopy(self.reference_evidence))
        if 'turn_id' in case:
            case['turn_id'] = self.turn_id
        return without_failure_labels(case)

    def record_classification(self, model: str, prediction: dict, audit: list[dict]):
        """Keep each classifier model's prediction and audit together on this turn."""
        if not isinstance(model, str) or not model.strip():
            raise ValueError('classification model must be a nonempty string')
        if not isinstance(prediction, dict) or not isinstance(audit, list):
            raise ValueError('prediction must be an object and audit must be a list')
        self.classifications[model] = {'prediction': copy.deepcopy(prediction), 'audit': copy.deepcopy(audit)}

    def to_dict(self) -> dict[str, Any]:
        return {'turn_id': self.turn_id, 'model': self.model,
                'case_input': self.to_case_input(), 'classifications': copy.deepcopy(self.classifications)}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> ConversationTurn:
        turn = cls.from_case_input(record['case_input'])
        # Run the same validation as direct construction for persisted metadata.
        turn_id, model = record['turn_id'], record.get('model')
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError('turn_id must be a nonempty string')
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError('model must be a nonempty string or None')
        turn.turn_id, turn.model = turn_id, model
        records = record.get('classifications', {})
        if not isinstance(records, dict):
            raise ValueError('classifications must be an object')
        for name, classification in records.items():
            turn.record_classification(name, classification['prediction'], classification['audit'])
        return turn


@dataclass
class Conversation:
    """An ordered collection of turns, with explicit JSON save/load."""

    conversation_id: str
    turns: list[ConversationTurn] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.conversation_id, str) or not self.conversation_id.strip():
            raise ValueError('conversation_id must be a nonempty string')
        turns = list(self.turns)
        self.turns = []
        for turn in turns:
            self.add_turn(turn)

    def add_turn(self, turn: ConversationTurn):
        if not isinstance(turn, ConversationTurn):
            raise TypeError('turn must be a ConversationTurn')
        if any(existing.turn_id == turn.turn_id for existing in self.turns):
            raise ValueError(f'Duplicate turn ID: {turn.turn_id}')
        self.turns.append(turn)

    def to_dict(self) -> dict[str, Any]:
        return {'schema_version': 1, 'conversation_id': self.conversation_id,
                'turns': [turn.to_dict() for turn in self.turns]}

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Conversation:
        if record.get('schema_version') != 1:
            raise ValueError('Unsupported conversation schema version')
        return cls(record['conversation_id'], [ConversationTurn.from_dict(turn) for turn in record['turns']])

    def save(self, path: str | Path):
        from failure_classification.errorclassification.data.case_io import write_json
        write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Conversation:
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
