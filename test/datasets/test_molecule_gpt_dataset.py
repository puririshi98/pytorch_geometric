import time
from typing import Any, List

import pytest
import requests

from torch_geometric.datasets import MoleculeGPTDataset
from torch_geometric.datasets.molecule_gpt_dataset import _get_pubchem_json
from torch_geometric.testing import withPackage

BUSY = {
    'Fault': {
        'Code': 'PUGVIEW.ServerBusy',
        'Message': 'Too many requests or server too busy',
    }
}
NOT_FOUND = {'Fault': {'Code': 'PUGVIEW.NotFound', 'Message': 'No data'}}
PAGE = {'Annotations': {'Page': 1, 'Annotation': []}}


class _Response:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self.ok = status_code < 400
        self.reason = 'OK' if self.ok else 'Error'
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, str):
            raise ValueError('Expecting value')
        return self._payload


def _mock_pubchem(monkeypatch, responses: List[Any]):
    calls: List[str] = []
    sleeps: List[float] = []

    def get(url, timeout):
        calls.append(url)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(requests, 'get', get)
    monkeypatch.setattr(time, 'sleep', sleeps.append)
    return calls, sleeps


def test_get_pubchem_json_retries_transient_errors(monkeypatch):
    calls, sleeps = _mock_pubchem(monkeypatch, [
        _Response(503, BUSY),
        requests.ConnectionError('conn reset'),
        requests.ReadTimeout('read timed out'),
        requests.exceptions.ChunkedEncodingError('IncompleteRead'),
        requests.exceptions.ContentDecodingError('bad gzip'),
        _Response(502, '<html>Bad Gateway</html>'),
        _Response(200, PAGE),
    ])
    out = _get_pubchem_json('url', num_retries=6, delay=1.0)
    assert out == PAGE
    assert calls == ['url'] * 7
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def test_get_pubchem_json_raises_after_retries(monkeypatch):
    calls, sleeps = _mock_pubchem(monkeypatch, [_Response(503, BUSY)] * 3)
    with pytest.raises(RuntimeError, match='PUGVIEW.ServerBusy.*3 attempts'):
        _get_pubchem_json('url', num_retries=2)
    assert len(calls) == 3
    assert len(sleeps) == 2

    calls, _ = _mock_pubchem(monkeypatch, [])
    with pytest.raises(ValueError, match='non-negative'):
        _get_pubchem_json('url', num_retries=-1)
    assert calls == []


@pytest.mark.parametrize('response', [
    _Response(404, NOT_FOUND),
    _Response(400, '<html>Bad Request</html>'),
    _Response(200, NOT_FOUND),
])
def test_get_pubchem_json_fails_fast(monkeypatch, response):
    calls, sleeps = _mock_pubchem(monkeypatch, [response])
    match = f'HTTP {response.status_code} .*after 1 attempt$'
    with pytest.raises(RuntimeError, match=match):
        _get_pubchem_json('url')
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize('payload', [None, {}, [1, 2, 3], '{"Annot'])
def test_get_pubchem_json_retries_malformed_body(monkeypatch, payload):
    # A `2xx` without a JSON object body is a truncated/garbled transfer:
    calls, sleeps = _mock_pubchem(monkeypatch, [
        _Response(200, payload),
        _Response(200, PAGE),
    ])
    out = _get_pubchem_json('url', delay=1.0)
    assert out == PAGE
    assert len(calls) == 2
    assert sleeps == [1.0]

    calls, _ = _mock_pubchem(monkeypatch, [_Response(200, payload)] * 2)
    with pytest.raises(RuntimeError, match='HTTP 200 .*malformed body.*'
                       '2 attempts'):
        _get_pubchem_json('url', num_retries=1)
    assert len(calls) == 2


@pytest.mark.parametrize('payload', [None, [1, 2], 'not json', {'Fault': 'x'}])
def test_get_pubchem_json_non_dict_error_body(monkeypatch, payload):
    calls, _ = _mock_pubchem(monkeypatch, [_Response(503, payload)] * 2)
    with pytest.raises(RuntimeError, match='HTTP 503 .*2 attempts'):
        _get_pubchem_json('url', num_retries=1)
    assert len(calls) == 2


@pytest.mark.dataset
@withPackage('transformers', 'sentencepiece', 'accelerate', 'rdkit')
def test_molecule_gpt_dataset():
    dataset = MoleculeGPTDataset(
        root='./data/MoleculeGPT',
        num_units=10,
    )
    assert str(dataset) == f'MoleculeGPTDataset({len(dataset)})'
    assert dataset.num_edge_features == 4
    assert dataset.num_node_features == 6
