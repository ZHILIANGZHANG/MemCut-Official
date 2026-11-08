"""Original native messages through the shared receipted, no-retry transport."""
import copy
import json


class NativeAbort(BaseException):
    """Cross upstream `except Exception: fallback` boundaries, then stop the run.

    This is converted to RuntimeError at the adapter boundary. A provider/schema
    failure must not become an empty bank, empty search, or a made-up query.
    """


def decode_json(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        lines = text.splitlines()
        if lines[0] not in ('```', '```json'):
            raise ValueError('Unsupported JSON fence')
        text = '\n'.join(lines[1:-1])
    return json.loads(text)


def validate_native_result(value, system):
    def strings(key, nonempty=False):
        rows = value.get(key)
        if not isinstance(rows, list) or (nonempty and not rows) or any(not isinstance(x, str) or not x.strip() for x in rows):
            raise ValueError(f'Invalid native retrieval {key}')
    if not isinstance(value, dict):
        raise ValueError('Native retrieval requires a JSON object')
    if 'information requirement analyst' in system:
        strings('key_entities')
        strings('relationships')
        rows = value.get('required_info')
        if not isinstance(rows, list) or not rows or any(
                not isinstance(r, dict) or any(not isinstance(r.get(k), str) or not r[k].strip()
                for k in ('info_type', 'description', 'priority')) for r in rows):
            raise ValueError('Missing native information requirements')
        if not isinstance(value.get('question_type'), str) or type(value.get('minimal_queries_needed')) is not int or value['minimal_queries_needed'] < 1:
            raise ValueError('Invalid native information plan')
    elif 'query analysis assistant' in system:
        for key in ('keywords', 'persons', 'entities'):
            strings(key)
        for key in ('time_expression', 'location'):
            if key not in value or (value[key] is not None and not isinstance(value[key], str)):
                raise ValueError(f'Invalid native query field: {key}')
    elif 'query generation specialist' in system:
        strings('queries', True)
    elif 'information completeness evaluator' in system:
        if value.get('assessment') not in ('complete', 'incomplete'):
            raise ValueError('Invalid completeness assessment')
    elif 'missing information query generator' in system:
        strings('targeted_queries')
        if len(value['targeted_queries']) > 3:
            raise ValueError('Missing-information query list exceeds native prompt limit')
    else:
        raise ValueError('Unexpected native retrieval request type')


class NativeClient:
    def __init__(self, completions, model, *, max_tokens, extraction=False):
        self.completions, self.model = completions, model
        self.max_tokens, self.extraction = max_tokens, extraction
        self.requests = []

    def chat_completion(self, messages, temperature=None, response_format=None, **kwargs):
        try:
            if kwargs:
                raise ValueError('Unexpected upstream completion parameters')
            request = {'model': self.model, 'messages': copy.deepcopy(messages),
                       'temperature': 0, 'max_tokens': self.max_tokens}
            if response_format:
                request['response_format'] = response_format
            response = self.completions.complete(request)
            self.requests.append({'request_id': response['request_id'], 'request': request})
            value = decode_json(response['text'])
            if not self.extraction:
                validate_native_result(value, messages[0]['content'])
            return response['text']
        except Exception as exc:
            raise NativeAbort(str(exc)) from exc

    @staticmethod
    def extract_json(text):
        try:
            return decode_json(text)
        except Exception as exc:
            raise NativeAbort(str(exc)) from exc
