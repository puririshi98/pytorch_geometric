import pytest

from torch_geometric.datasets import MoleculeGPTDataset, molecule_gpt_dataset
from torch_geometric.testing import withPackage


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.ok = status_code < 400
        self.reason = 'OK' if self.ok else 'Service Unavailable'
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_get_pubchem_json_retries_on_server_busy(monkeypatch):
    fault = {
        'Fault': {
            'Code': 'PUGVIEW.ServerBusy',
            'Message': 'Too many requests or server too busy',
        }
    }
    page = {'Annotations': {'Page': 1, 'Annotation': []}}
    responses = [
        _Response(503, fault),
        _Response(503, fault),
        _Response(200, page)
    ]
    urls = []

    def get(url):
        urls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(molecule_gpt_dataset.requests, 'get', get)
    monkeypatch.setattr(molecule_gpt_dataset.time, 'sleep', lambda _: None)

    out = molecule_gpt_dataset._get_pubchem_json('url', num_retries=2)
    assert out == page
    assert urls == ['url'] * 3

    responses[:] = [_Response(503, fault)] * 3
    with pytest.raises(RuntimeError, match='PUGVIEW.ServerBusy'):
        molecule_gpt_dataset._get_pubchem_json('url', num_retries=2)


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
