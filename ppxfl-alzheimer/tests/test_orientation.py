"""Tests for the orientation search and for scan selection."""

import numpy as np

from src.orientation import (
    FLIPS,
    PERMUTATIONS,
    apply_orientation,
    downsample,
    group_key,
    load_table,
    shape_cube,
    shortlist_by_shape,
)
from src.preprocess_v3 import ScanRecord, _acq_date_of, _series_of, _series_rank, select_one_per_subject


def test_there_are_forty_eight_candidate_orientations():
    assert len(PERMUTATIONS) * len(FLIPS) == 48


def test_identity_orientation_returns_the_same_array():
    volume = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert np.array_equal(apply_orientation(volume, (0, 1, 2), (False,) * 3), volume)


def test_permutation_transposes_the_axes():
    volume = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert apply_orientation(volume, (2, 0, 1), (False,) * 3).shape == (4, 2, 3)


def test_flip_reverses_the_requested_axis():
    volume = np.arange(8, dtype=float).reshape(2, 2, 2)
    flipped = apply_orientation(volume, (0, 1, 2), (True, False, False))
    assert np.array_equal(flipped, volume[::-1])


def test_every_candidate_preserves_the_voxel_multiset():
    volume = np.arange(60, dtype=float).reshape(3, 4, 5)
    for permutation in PERMUTATIONS:
        for flips in FLIPS:
            candidate = apply_orientation(volume, permutation, flips)
            assert sorted(candidate.ravel()) == sorted(volume.ravel())


def test_downsample_by_one_is_a_no_op():
    volume = np.random.default_rng(0).random((6, 6, 6))
    assert np.array_equal(downsample(volume, 1), volume)


def test_downsample_averages_blocks():
    volume = np.ones((6, 6, 6))
    small = downsample(volume, 3)
    assert small.shape == (2, 2, 2)
    assert np.allclose(small, 1.0)


def test_downsample_trims_a_ragged_edge():
    volume = np.ones((7, 7, 7))
    assert downsample(volume, 3).shape == (2, 2, 2)


def test_downsample_leaves_volumes_smaller_than_the_factor_alone():
    volume = np.ones((2, 2, 2))
    assert downsample(volume, 3).shape == (2, 2, 2)


def test_shape_cube_has_a_fixed_size_regardless_of_input_shape():
    rng = np.random.default_rng(0)
    for shape in [(20, 30, 40), (64, 64, 64), (13, 41, 27)]:
        volume = rng.random(shape) + 1.0
        assert shape_cube(volume, size=16).shape == (16, 16, 16)


def test_shape_cube_of_an_empty_volume_is_zero():
    assert shape_cube(np.zeros((10, 10, 10)), size=8).sum() == 0.0


def _graded_blob(shape, corner, size):
    """A blob whose intensity varies along one axis, so it has a front and a back."""
    volume = np.zeros(shape)
    ramp = np.linspace(0.2, 1.0, size)
    block = np.broadcast_to(ramp[:, None, None], (size, size, size))
    slices = tuple(slice(c, c + size) for c in corner)
    volume[slices] = block
    return volume


def test_shape_cube_is_scale_and_position_invariant():
    """The same blob, moved and rescaled inside the array, gives the same cube."""
    small = _graded_blob((40, 40, 40), (10, 12, 14), 10)
    large = _graded_blob((80, 80, 80), (40, 20, 50), 20)
    assert np.corrcoef(shape_cube(small, 16).ravel(),
                       shape_cube(large, 16).ravel())[0, 1] > 0.99


def test_shape_cube_distinguishes_a_flipped_blob():
    """Reversing the intensity ramp must change the cube, or nothing can be scored."""
    blob = _graded_blob((40, 40, 40), (10, 12, 14), 10)
    flipped = apply_orientation(blob, (0, 1, 2), (True, False, False))
    assert np.corrcoef(shape_cube(blob, 16).ravel(),
                       shape_cube(flipped, 16).ravel())[0, 1] < 0.5


def test_shortlist_returns_the_requested_number_of_candidates():
    rng = np.random.default_rng(1)
    volume = rng.random((24, 24, 24))
    reference = shape_cube(rng.random((24, 24, 24)), size=48)
    assert len(shortlist_by_shape(volume, reference, 7)) == 7


def test_shortlist_is_ordered_by_descending_score():
    rng = np.random.default_rng(2)
    volume = rng.random((24, 24, 24))
    reference = shape_cube(volume, size=48)
    scores = [s for s, _, _ in shortlist_by_shape(volume, reference, 10)]
    assert scores == sorted(scores, reverse=True)


def test_group_key_normalises_the_series_family():
    assert group_key("Accelerated_SAG_IR-SPGR", (256, 256, 196)) == "IR-SPGR|256x256x196"
    assert group_key("MPRAGE_repeat", (256, 240, 160)) == "MPRAGE|256x240x160"
    assert group_key("MT1__GradWarp__N3m", (1, 2, 3)) == "MT1__GRADWARP__N3M|1x2x3"


def test_load_table_of_a_missing_file_is_empty(tmp_path):
    assert load_table(str(tmp_path / "absent.json")) == {}


def test_load_table_parses_permutation_and_flips(tmp_path):
    import json

    path = tmp_path / "table.json"
    path.write_text(json.dumps({
        "MPRAGE|1x2x3": {"permutation": [2, 0, 1], "flips": [False, True, False]}
    }), encoding="utf-8")
    table = load_table(str(path))
    assert table["MPRAGE|1x2x3"] == ((2, 0, 1), (False, True, False))


def _record(subject, series, date, scan_id):
    return ScanRecord(subject_id=subject, site_id=subject.split("_")[0], klass="AD",
                      label=2, scan_id=scan_id, series=series, acq_date=date,
                      source_path=f"/data/{subject}/{series}/{scan_id}.nii")


def test_series_rank_prefers_more_processed_products():
    assert (_series_rank("MPR__GradWarp__B1_Correction__N3__Scaled")
            > _series_rank("MPR__GradWarp__B1_Correction")
            > _series_rank("MPR__GradWarp")
            > _series_rank("MPRAGE"))


def test_selection_prefers_the_most_processed_series():
    records = [
        _record("002_S_0001", "MPRAGE", "2006-01-01", "I1"),
        _record("002_S_0001", "MPR__GradWarp__B1_Correction__N3", "2006-01-01", "I2"),
    ]
    chosen = select_one_per_subject(records)
    assert len(chosen) == 1
    assert chosen[0].scan_id == "I2"


def test_selection_prefers_the_earliest_visit_at_equal_processing():
    records = [
        _record("002_S_0001", "MPRAGE", "2008-05-05", "I9"),
        _record("002_S_0001", "MPRAGE", "2006-01-01", "I1"),
    ]
    assert select_one_per_subject(records)[0].scan_id == "I1"


def test_selection_keeps_one_scan_per_subject():
    records = [
        _record("002_S_0001", "MPRAGE", "2006-01-01", "I1"),
        _record("002_S_0001", "MPRAGE", "2007-01-01", "I2"),
        _record("003_S_0002", "MPRAGE", "2006-01-01", "I3"),
    ]
    chosen = select_one_per_subject(records)
    assert sorted(r.subject_id for r in chosen) == ["002_S_0001", "003_S_0002"]


def test_acquisition_date_is_read_from_the_path():
    path = "/data/002_S_0001/MPRAGE/2006-06-01_20_04_45.0/I1/scan.nii"
    assert _acq_date_of(path) == "2006-06-01"


def test_acquisition_date_falls_back_when_absent():
    assert _acq_date_of("/data/no/date/here.nii") == "9999-99-99"


def test_series_is_the_directory_after_the_subject():
    import os

    path = os.path.join("data", "AD-150", "ADNI", "002_S_0619",
                        "MPR-R__GradWarp", "2006-06-01_20_04_45.0", "I1", "s.nii")
    assert _series_of(path) == "MPR-R__GradWarp"


def test_series_of_a_path_without_a_subject_is_empty():
    assert _series_of("/tmp/scan.nii") == ""


def test_shortlist_of_an_empty_volume_has_no_candidates():
    """Every candidate of an all-zero volume is featureless and is dropped."""
    reference = shape_cube(np.random.default_rng(3).random((24, 24, 24)), size=48)
    assert shortlist_by_shape(np.zeros((24, 24, 24)), reference, 5) == []
