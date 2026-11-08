"""Shared synthetic fixtures; native integration tests remain in the local workspace."""
import asyncio

import copy

import hashlib

import json

import os

from pathlib import Path

import re

import tempfile

import unittest

from unittest.mock import patch

from memcut.native_memory.memorybank import make_memorybank

from memcut.native_memory.contract import digest

from memcut.source_anchored_v2 import PROTOCOL

from memcut.source_anchored_v2.memorybank import SourceMemoryBank, address

from memcut.source_anchored_v2.provenance import diff_objects

from memcut.source_anchored_v2.completions import CachedCompletions, native_summary_request

from memcut.source_anchored_v2.candidates import build_pool, selector_input, rank_s2

from memcut.source_anchored_v2.evaluation import contexts, evaluate, summarize, score_response

from memcut.source_anchored_v2 import runner

def run(coro):
    return asyncio.run(coro)

def episode():
    sessions = []
    for day in range(1, 3):
        turns = []
        for i in range(12):
            for role in ('user', 'assistant'):
                j = len(turns)+1
                word = f'SECRET_{day}_{i}' if role == 'user' else f'KEEP_{day}_{i}'
                turns.append({'id': f'S{day}:T{j}', 'role': role, 'speaker': 'Mira' if role == 'user' else 'AI',
                              'text': f'The stored record is {word}. ' + 'Archive paperwork. '*10})
        sessions.append({'id': f'S{day}', 'timestamp': f'2026-09-0{day}', 'turns': turns})
    return {'id': 'fixture', 'subject': 'Mira', 'sessions': sessions}

def claim():
    return {**runner.normalize_claim({'claim_id': 'c1', 'stem': 'Which is correct?',
        'choices': [{'id': 'right', 'text': 'Correct option'}] +
                   [{'id': f'wrong{i}', 'text': f'Wrong option {i}'} for i in range(4)],
        'target_choice_id': 'right'}),
        'evidence_turns': ['S1:T1'], 'disclosure_turns': ['S1:T3'], 'distractor_turns': ['S1:T2'],
        'deletion_request': 'Forget the private archived record.'}
