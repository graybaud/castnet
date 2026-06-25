"""Tests for castnet.hardware.neuromorphic."""
from domain.hardware.neuromorphic import compare_castnet, EXISTING_CHIPS

class TestCompareCastnet:
    def test_basic(self):
        result = compare_castnet(25600, 15_300_000, "130nm")
        assert "castnet" in result
        assert "comparison" in result
        assert len(result["comparison"]) == 4  # 3 existing + CastNet

    def test_castnet_entry(self):
        result = compare_castnet(25600, 15_300_000, "130nm")
        c = result["castnet"]
        assert c["neurons"] == 25600
        assert c["synapses"] == 15_300_000
        assert c["programmable"] == False

    def test_existing_chips_present(self):
        result = compare_castnet(100, 1000, "130nm")
        names = [c["name"] for c in result["comparison"]]
        assert any("Loihi" in n for n in names)
        assert any("TrueNorth" in n for n in names)
        assert any("SpiNNaker" in n for n in names)
        assert any("CastNet" in n for n in names)

    def test_synapse_density_positive(self):
        result = compare_castnet(25600, 15_300_000, "130nm")
        for c in result["comparison"]:
            assert c["synapse_density_per_mm2"] > 0
