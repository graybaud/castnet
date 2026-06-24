import json, os, tempfile
from domain.visualization.graph_export import prepare_graph_for_export, export_graph_json

class TestPrepareGraphForExport:
    def test_basic(self):
        d = prepare_graph_for_export([0,1],[1,2],[0.5,-0.3],[True,False],total_nodes=3)
        assert len(d["nodes"]) == 3
        assert len(d["edges"]) == 2
    def test_auto_nodes(self):
        d = prepare_graph_for_export([0,2],[1,3],[0.5,0.3])
        assert len(d["nodes"]) == 4
    def test_max_nodes(self):
        d = prepare_graph_for_export(list(range(100)),[i+1 for i in range(100)],[0.1]*100,max_nodes=50,total_nodes=101)
        assert len(d["nodes"]) == 50
    def test_metrics(self):
        d = prepare_graph_for_export([0],[1],[0.5],total_nodes=2)
        assert d["metrics"]["total_nodes"] == 2

class TestExport:
    def test_export(self):
        d = prepare_graph_for_export([0,1],[1,2],[0.5,0.3],total_nodes=3)
        with tempfile.NamedTemporaryFile(mode="w",suffix=".json",delete=False) as f: p=f.name
        try:
            assert export_graph_json(d,p) == p
            assert os.path.exists(p)
        finally:
            os.unlink(p)
