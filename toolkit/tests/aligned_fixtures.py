"""Synthetic native-integration fixtures; no API, model downloads or audit imports."""
import re
import json
import numpy as np
from memcut.source_anchored_v2.completions import CachedCompletions


class Embedding:
    dimension = 384

    def __init__(self):
        self.inputs = []

    def encode_documents(self, texts):
        self.inputs.extend(texts)
        vectors = []
        for text in texts:
            match = re.search(r'Rank (\d+)', text)
            n = int(match[1]) if match else 0
            v = np.zeros(self.dimension, dtype=np.float32)
            v[:2] = [1/(n+1), (1-(1/(n+1))**2)**.5]
            vectors.append(v)
        return np.asarray(vectors)

    def encode_single(self, text, is_query=False):
        return self.encode_documents([text])[0]

def fact(n, source_id):
    return {'lossless_restatement': f'Rank {n}: Mira records item{n} at Harbor.',
        'keywords': [f'keywordonly{n}'], 'timestamp': '2026-01-01T12:00:00',
        'location': 'Harbor', 'persons': ['Mira'], 'entities': [f'item{n}'], 'topic': 'Inventory',
        'source_ids': [source_id], 'source_memory_ids': []}

def response(**request):
    system, body = request['messages'][0]['content'], request['messages'][-1]['content']
    assert request['temperature'] == 0
    if 'professional information extraction' in system:
        lines = re.findall(r'\[turn (\d+)\].*?Rank (\d+):', body)
        content = [fact(int(n), int(s)) for s,n in lines]
        if lines and lines[0][1] == '1':
            content.append({**fact(1,1), 'lossless_restatement': 'Rank 1: Mira combined item1 and item2.',
                            'source_ids': [1,2]})
        assert request['max_tokens'] == 8192
    elif 'information requirement analyst' in system:
        content = {'question_type': 'factual', 'key_entities': ['Mira'],
            'required_info': [{'info_type': 'fact', 'description': 'inventory', 'priority': 'high'}],
            'relationships': [], 'minimal_queries_needed': 2}
    elif 'query generation specialist' in system:
        content = {'reasoning': 'Two views.', 'queries': ['Rank 0 query', 'Rank 75 query']}
    elif 'query analysis assistant' in system:
        content = {'keywords': ['item82'], 'persons': ['Mira'], 'time_expression': None,
                   'location': 'Harbor', 'entities': ['item70']}
    elif 'information completeness evaluator' in system:
        content = {'assessment': 'complete', 'reasoning': 'Fixture complete.', 'coverage_percentage': 100}
    elif 'missing information query generator' in system:
        content = {'targeted_queries': ['item82'], 'missing_analysis': 'One gap.'}
    elif 'Score candidate turns for a targeted memory deletion' in system:
        payload = json.loads(body)
        assert set(payload) == {'deletion_request','objects','candidates'}
        content = {'scores': [{'id': row['id'], 'score': 10*i/max(1,len(payload['candidates'])-1),
                              'reason': 'Fixture reverse ranking.'} for i,row in enumerate(payload['candidates'])]}
    else:
        assert 'multiple-choice' in system and request['max_tokens'] == 2048
        label = re.search(r'^([a-e])\. Known option$', body, re.M)[1]
        content = {'reason': 'Fixture answer.', 'answer': label}
    return {'choices': [{'finish_reason':'stop','message':{'content':json.dumps(content)}}],
            'usage': {'prompt_tokens':10, 'completion_tokens':5}}

class FixtureGateway:
    def __init__(self, folder, create=response):
        self.cache = CachedCompletions(folder/'requests', create, endpoint='offline-fixture')
    def for_role(self, role):
        return self.cache
    @property
    def calls(self):
        return self.cache.calls
    def close(self):
        pass

def source_turns(n=4):
    return {f't{i}': {'turn_id': f't{i}', 'session_id': 's1', 'role': 'user' if i%2 else 'assistant',
                     'speaker': 'Alice' if i%2 else 'Bob', 'timestamp': f'2026-01-01T12:00:{i:02}',
                     'text': f'The receipt code is CODE{i:02}. Please retain this receipt code.'}
            for i in range(1, n+1)}
