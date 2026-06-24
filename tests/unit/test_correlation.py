import pytest
from domain.analysis.correlation import pearson_correlation, mask_overlap, overlap_matrix, interpret_correlation, orthogonality_score

class TestPearson:
    def test_perfect_pos(self): assert pearson_correlation([1,2,3,4],[1,2,3,4]) == pytest.approx(1.0)
    def test_perfect_neg(self): assert pearson_correlation([1,2,3,4],[4,3,2,1]) == pytest.approx(-1.0)
    def test_uncorrelated(self): assert pearson_correlation([1,2,3,4],[5,5,5,5]) == 0.0
    def test_empty(self): assert pearson_correlation([],[]) == 0.0

class TestMaskOverlap:
    def test_identical(self): assert mask_overlap({0:1,1:1},{0:1,1:1}) == 1.0
    def test_disjoint(self): assert mask_overlap({0:1,1:0},{0:0,1:1}) == 0.0
    def test_half(self): assert mask_overlap({0:1,1:1,2:0},{0:1,1:0,2:1}) == 0.5

class TestOverlapMatrix:
    def test_3x3(self):
        m = {"A":{0:1,1:1,2:0},"B":{0:1,1:0,2:1},"C":{0:0,1:1,2:1}}
        mat = overlap_matrix(m,["A","B","C"])
        assert mat[0][0] == 1.0
        assert mat[1][1] == 1.0

class TestInterpret:
    def test_orth(self): assert interpret_correlation(0.1) == "ORTHOGONAL"
    def test_partial(self): assert interpret_correlation(0.4) == "PARTIALLY CORRELATED"
    def test_high(self): assert interpret_correlation(0.8) == "HIGHLY CORRELATED"

class TestOrthoScore:
    def test_perfect(self): assert orthogonality_score([0,0,0]) == 100.0
    def test_full_overlap(self): assert orthogonality_score([1,1,1]) == 0.0
    def test_empty(self): assert orthogonality_score([]) == 100.0
