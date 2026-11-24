"""The cell grid, which has to be complete in every state for the run report to mean anything."""

import datetime
import logging

import pytest
from gbstats.models.statistics import compute_theta_regression_adjusted_ratio

from highwind import gbstats_compute, metrics, units
from highwind.discovery import Experiment, Window

WINDOWS = [
    Window(label="cumu:1", start=0, end=6, kind="cumulative"),
    Window(label="week:1", start=0, end=6, kind="disjoint"),
    Window(label="week:2", start=7, end=13, kind="disjoint"),
]

CONTINUOUS = metrics.Metric(
    name="active_hours",
    source="clients_daily",
    reducer=None,
    window_rules=(metrics.CUMULATIVE_WEEKLY,),
)
BINARY = metrics.Metric(
    name="retained_dau",
    source="active_users",
    reducer=None,
    window_rules=(metrics.DISJOINT_WEEKLY,),
)

EXPERIMENT = Experiment(
    slug="a-slug",
    start_date=datetime.date(2026, 7, 1),
    end_date=None,
    reference_branch="control",
    treatment_branches=("treatment-a", "treatment-b"),
    unit=units.resolve("firefox_desktop", "normandy_id"),
)


def branch(units, mean, second_moment, cross_moment):
    """One branch's aggregates, for units whose covariate mean equals their metric mean."""
    return dict(
        n=units,
        sum=units * mean,
        sum_squares=units * second_moment,
        pre_sum=units * 1.0,
        pre_sum_squares=units * 2.0,
        sum_x_pre=units * cross_moment,
    )


def test_a_metric_only_gets_the_windows_of_the_families_it_declares():
    continuous = gbstats_compute.windows_for_metric(CONTINUOUS, WINDOWS)
    binary = gbstats_compute.windows_for_metric(BINARY, WINDOWS)

    assert [window.label for window in continuous] == ["cumu:1"]
    assert [window.label for window in binary] == ["week:1", "week:2"]


def test_the_grid_is_one_cell_per_metric_window_and_treatment_branch():
    results = gbstats_compute.compute_statistics(
        EXPERIMENT, [CONTINUOUS, BINARY], WINDOWS, cells={}
    )

    cells = {
        (result["metric"], result["window"], result["segment"], result["branch"])
        for result in results
    }
    assert len(results) == 6
    assert cells == {
        ("active_hours", "cumu:1", "all_enrolled", "treatment-a"),
        ("active_hours", "cumu:1", "all_enrolled", "treatment-b"),
        ("retained_dau", "week:1", "all_enrolled", "treatment-a"),
        ("retained_dau", "week:1", "all_enrolled", "treatment-b"),
        ("retained_dau", "week:2", "all_enrolled", "treatment-a"),
        ("retained_dau", "week:2", "all_enrolled", "treatment-b"),
    }


def test_a_cell_with_no_aggregates_is_not_started_rather_than_absent():
    results = gbstats_compute.compute_statistics(
        EXPERIMENT, [CONTINUOUS], WINDOWS, cells={}
    )

    assert {result["state"] for result in results} == {gbstats_compute.NOT_STARTED}
    assert all("point" not in result for result in results)


def test_a_failure_marks_every_declared_cell_an_error_so_the_run_can_see_it():
    results = gbstats_compute.compute_statistics(
        EXPERIMENT,
        [CONTINUOUS, BINARY],
        WINDOWS,
        cells={},
        failure=RuntimeError("the source query failed"),
    )

    assert len(results) == 6
    assert {result["state"] for result in results} == {gbstats_compute.ERROR}
    assert all("the source query failed" in result["error"] for result in results)


def test_a_segment_failure_marks_only_that_segments_cells_as_error():
    results = gbstats_compute.compute_statistics(
        EXPERIMENT,
        [CONTINUOUS],
        WINDOWS,
        cells={},
        segments=("all_enrolled", "regular_user"),
        segment_failures={"regular_user": RuntimeError("the segment source failed")},
    )

    states_by_segment = {result["segment"]: result["state"] for result in results}
    assert states_by_segment["all_enrolled"] == gbstats_compute.NOT_STARTED
    assert states_by_segment["regular_user"] == gbstats_compute.ERROR
    regular_user_errors = [
        result["error"] for result in results if result["segment"] == "regular_user"
    ]
    assert all("the segment source failed" in error for error in regular_user_errors)


def test_a_run_wide_failure_overrides_a_segments_own_failure_message():
    # A shared-scan failure already covers every segment's cells, so it is the one recorded even
    # when a segment also has its own `segment_failures` entry.
    results = gbstats_compute.compute_statistics(
        EXPERIMENT,
        [CONTINUOUS],
        WINDOWS,
        cells={},
        failure=RuntimeError("the cohort query failed"),
        segments=("all_enrolled", "regular_user"),
        segment_failures={"regular_user": RuntimeError("the segment source failed")},
    )

    assert {result["state"] for result in results} == {gbstats_compute.ERROR}
    assert all("the cohort query failed" in result["error"] for result in results)


def test_a_branch_below_the_minimum_units_is_insufficient_data_not_an_error():
    cells = {
        ("active_hours", "cumu:1", "all_enrolled", "control"): branch(400, 1.0, 2.0, 1.5),
        ("active_hours", "cumu:1", "all_enrolled", "treatment-a"): branch(1, 1.0, 2.0, 1.5),
    }
    results = {
        result["branch"]: result
        for result in gbstats_compute.compute_statistics(
            EXPERIMENT, [CONTINUOUS], WINDOWS, cells
        )
    }

    assert results["treatment-a"]["state"] == gbstats_compute.INSUFFICIENT_DATA
    assert results["treatment-a"]["n_reference"] == 400
    assert results["treatment-a"]["n_treatment"] == 1
    # Nothing was computed for this branch at all, which is a different state again.
    assert results["treatment-b"]["state"] == gbstats_compute.NOT_STARTED


def test_a_populated_pair_produces_a_relative_interval_around_the_difference():
    cells = {
        ("active_hours", "cumu:1", "all_enrolled", "control"): branch(1000, 1.0, 2.0, 1.5),
        ("active_hours", "cumu:1", "all_enrolled", "treatment-a"): branch(1000, 1.1, 2.42, 1.65),
    }
    results = {
        result["branch"]: result
        for result in gbstats_compute.compute_statistics(
            EXPERIMENT, [CONTINUOUS], WINDOWS, cells
        )
    }
    treatment = results["treatment-a"]

    assert treatment["state"] in (gbstats_compute.FORMING, gbstats_compute.CONFIDENT)
    assert treatment["point"] == pytest.approx(10.0, abs=1.0)
    assert treatment["lower"] < treatment["point"] < treatment["upper"]
    assert treatment["theta"] > 0


def test_the_absolute_interval_is_the_relative_one_rescaled_by_the_reference_mean():
    cells = {
        ("active_hours", "cumu:1", "all_enrolled", "control"): branch(1000, 1.0, 2.0, 1.5),
        ("active_hours", "cumu:1", "all_enrolled", "treatment-a"): branch(1000, 1.1, 2.42, 1.65),
    }
    results = {
        result["branch"]: result
        for result in gbstats_compute.compute_statistics(
            EXPERIMENT, [CONTINUOUS], WINDOWS, cells
        )
    }
    treatment = results["treatment-a"]
    absolute = treatment["absolute"]

    assert absolute["point"] == pytest.approx(0.1)
    for bound in ("point", "lower", "upper"):
        assert absolute[bound] == pytest.approx(treatment[bound] / 100 * 1.0)


@pytest.mark.parametrize(
    "reference_mean, treatment_mean, units",
    [
        (1.0, 1.1, 1000),
        (2.0, 1.7, 1000),
        (-2.0, -2.3, 1000),
        (5.0, 5.0, 400),
        (1.0, 1.0823, 3000),
        (1.0, 1.0824, 3000),
        (1.0, 1.1, 20),
    ],
)
def test_the_absolute_and_relative_intervals_exclude_zero_identically(
    reference_mean, treatment_mean, units
):
    reference = branch(units, reference_mean, reference_mean**2 + 1.0, reference_mean)
    candidate = branch(units, treatment_mean, treatment_mean**2 + 1.0, treatment_mean)

    interval = gbstats_compute.sequential_interval(reference, candidate, 0.0)
    absolute = interval["absolute"]

    for bound in ("point", "lower", "upper"):
        assert absolute[bound] == pytest.approx(interval[bound] / 100 * abs(reference_mean))
    assert (absolute["lower"] > 0) == (interval["lower"] > 0)
    assert (absolute["upper"] < 0) == (interval["upper"] < 0)
    assert absolute["lower"] <= absolute["point"] <= absolute["upper"]


def test_the_marginal_cases_straddle_zero_the_same_way_on_both_scales():
    verdicts = []
    for treatment_mean in (1.0823, 1.0824):
        interval = gbstats_compute.sequential_interval(
            branch(3000, 1.0, 2.0, 1.0),
            branch(3000, treatment_mean, treatment_mean**2 + 1.0, treatment_mean),
            0.0,
        )
        verdicts.append((interval["lower"] > 0, interval["absolute"]["lower"] > 0))

    assert all(relative == absolute for relative, absolute in verdicts)
    assert {relative for relative, _ in verdicts} == {False, True}


def test_cells_with_no_interval_carry_no_absolute_interval_either():
    cells = {
        ("active_hours", "cumu:1", "all_enrolled", "control"): branch(400, 1.0, 2.0, 1.5),
        ("active_hours", "cumu:1", "all_enrolled", "treatment-a"): branch(1, 1.0, 2.0, 1.5),
    }
    results = gbstats_compute.compute_statistics(EXPERIMENT, [CONTINUOUS], WINDOWS, cells)

    assert not any("absolute" in result for result in results)


def test_theta_is_zero_when_the_covariate_never_moves():
    flat = dict(n=40, sum=40.0, sum_squares=80.0, pre_sum=0.0, pre_sum_squares=0.0, sum_x_pre=0.0)

    assert gbstats_compute.pooled_theta([flat, flat]) == 0.0


def test_theta_is_the_slope_of_the_metric_on_the_covariate():
    # Two units per branch, each unit's value exactly twice its covariate, so the slope is 2.
    branches = [
        dict(n=2, sum=6.0, sum_squares=20.0, pre_sum=3.0, pre_sum_squares=5.0, sum_x_pre=10.0),
        dict(n=2, sum=14.0, sum_squares=100.0, pre_sum=7.0, pre_sum_squares=25.0, sum_x_pre=50.0),
    ]

    assert gbstats_compute.pooled_theta(branches) == pytest.approx(2.0)


def test_an_interval_clear_of_zero_is_what_separates_confident_from_forming():
    assert gbstats_compute.excludes_zero({"lower": 0.5, "upper": 2.0}) is True
    assert gbstats_compute.excludes_zero({"lower": -2.0, "upper": -0.5}) is True
    assert gbstats_compute.excludes_zero({"lower": -0.5, "upper": 2.0}) is False


# ----------------------------------------------------- the sequential tuning parameter ----


def test_the_tuning_parameter_reaches_the_test_on_the_scale_gbstats_reads_it_at():
    # gbstats multiplies the mixture variance it derives from this number by its own unit count,
    # which is the two branches summed. A parameter set on any other scale is silently mistuned, and
    # nothing downstream would say so, since it moves width rather than validity.
    test = gbstats_compute.build_t_test(
        gbstats_compute.adjusted(branch(6000, 1.0, 2.0, 1.5), 0.5),
        gbstats_compute.adjusted(branch(4000, 1.0, 2.0, 1.5), 0.5),
    )

    assert test.n == 10000
    assert test.sequential_tuning_parameter == 10000
    assert test.sequential_tuning_parameter != 5000  # gbstats' own default


def test_the_tuning_parameter_holds_still_while_a_cohort_grows():
    # The always-valid guarantee is a property of ONE confidence sequence fixed ahead of the data.
    # Retuning from each run's own unit count would read the narrowest member of a family of
    # sequences at every look, which none of them covers, so the parameter is quantised to hold
    # still from run to run as units mature.
    growing = [700_000, 800_000, 900_000, 1_000_000, 1_200_000]

    assert {gbstats_compute.tuning_parameter(count) for count in growing} == {1_000_000}


def test_the_tuning_parameter_tracks_the_unit_count_across_scales():
    # It has to hold still without going stale: a parameter far above the unit count widens the
    # interval by orders of magnitude, so a fixed constant is not an alternative to the grid.
    for count, expected in (
        (4, 10),
        (900, 1000),
        (20_000, 10_000),
        (40_000, 100_000),
        (60_000_000, 100_000_000),
    ):
        assert gbstats_compute.tuning_parameter(count) == expected

    assert gbstats_compute.tuning_parameter(0) == 1


# ---------------------------------------------------------- theta across every branch ----


def branch_moments(units, post_mean, post_variance, pre_mean, pre_variance, covariance):
    """One branch's six aggregates, written as the moments they encode.

    The branches differ in their covariate mean as well as their slope, which is the case CUPED
    exists for and the case where the choice of theta moves the point estimate rather than only the
    width.
    """
    return dict(
        n=units,
        sum=units * post_mean,
        sum_squares=units * post_mean**2 + (units - 1) * post_variance,
        pre_sum=units * pre_mean,
        pre_sum_squares=units * pre_mean**2 + (units - 1) * pre_variance,
        sum_x_pre=units * post_mean * pre_mean + (units - 1) * covariance,
    )


def three_branch_cells(window="cumu:1"):
    """One (metric, window) whose three branches have visibly different covariate slopes."""
    return {
        ("active_hours", window, "all_enrolled", "control"): branch_moments(
            1000, 1.0, 1.0, 1.0, 1.0, 0.5
        ),
        ("active_hours", window, "all_enrolled", "treatment-a"): branch_moments(
            1000, 1.5, 1.2, 1.3, 1.4, 0.9
        ),
        ("active_hours", window, "all_enrolled", "treatment-b"): branch_moments(
            1000, 0.9, 0.8, 0.7, 0.6, 0.2
        ),
    }


def test_every_comparison_of_one_metric_and_window_shares_one_theta():
    results = gbstats_compute.compute_statistics(
        EXPERIMENT, [CONTINUOUS], WINDOWS, three_branch_cells()
    )
    thetas = {result["branch"]: result["theta"] for result in results}

    assert set(thetas) == {"treatment-a", "treatment-b"}
    assert thetas["treatment-a"] == thetas["treatment-b"]


def test_three_branches_pool_a_different_theta_than_either_pair_would():
    # The assertion that pins the change: pooling over all branches is not the same number as
    # pooling over the pair being compared, so a run's intervals move rather than being relabelled.
    cells = three_branch_cells()
    reference = cells[("active_hours", "cumu:1", "all_enrolled", "control")]
    results = {
        result["branch"]: result
        for result in gbstats_compute.compute_statistics(
            EXPERIMENT, [CONTINUOUS], WINDOWS, cells
        )
    }

    for treatment in ("treatment-a", "treatment-b"):
        candidate = cells[("active_hours", "cumu:1", "all_enrolled", treatment)]
        pairwise = gbstats_compute.pooled_theta([reference, candidate])
        per_pair = gbstats_compute.sequential_interval(reference, candidate, pairwise)

        assert results[treatment]["theta"] != pytest.approx(pairwise, rel=1e-6)
        assert results[treatment]["point"] != pytest.approx(
            per_pair["point"], rel=1e-6
        )


def test_theta_pools_only_the_arms_that_reported_the_window():
    # A branch with no cell for this window contributes nothing rather than crashing the group.
    cells = three_branch_cells()
    del cells[("active_hours", "cumu:1", "all_enrolled", "treatment-b")]
    window = next(window for window in WINDOWS if window.label == "cumu:1")

    assert gbstats_compute.window_theta(
        EXPERIMENT, CONTINUOUS, window, cells
    ) == pytest.approx(
        gbstats_compute.pooled_theta(
            [
                cells[("active_hours", "cumu:1", "all_enrolled", "control")],
                cells[("active_hours", "cumu:1", "all_enrolled", "treatment-a")],
            ]
        )
    )
    assert gbstats_compute.window_theta(EXPERIMENT, CONTINUOUS, window, {}) == 0.0


def test_a_cell_whose_statistics_raise_is_an_error_with_a_log_record_naming_it(caplog):
    caplog.set_level(logging.ERROR)
    unusable = dict(branch(400, 1.0, 2.0, 1.5), sum_squares=None)
    cells = {
        ("active_hours", "cumu:1", "all_enrolled", "control"): branch(400, 1.0, 2.0, 1.5),
        ("active_hours", "cumu:1", "all_enrolled", "treatment-a"): unusable,
    }

    results = {
        result["branch"]: result
        for result in gbstats_compute.compute_statistics(
            EXPERIMENT, [CONTINUOUS], WINDOWS, cells
        )
    }

    assert results["treatment-a"]["state"] == gbstats_compute.ERROR
    assert results["treatment-a"]["error"].startswith("ValidationError")
    [record] = caplog.records
    assert record.experiment_slug == "a-slug"
    assert record.metric == "active_hours"
    assert record.segment == "all_enrolled"
    assert record.window == WINDOWS[0]
    assert record.exc_info[0].__name__ == "ValidationError"


# ------------------------------------------------ comparisons gbstats cannot estimate ----


def compute_pair(reference, candidate):
    """The treatment-a result for one (metric, window) holding only the given pair."""
    cells = {
        ("active_hours", "cumu:1", "all_enrolled", "control"): reference,
        ("active_hours", "cumu:1", "all_enrolled", "treatment-a"): candidate,
    }
    results = gbstats_compute.compute_statistics(EXPERIMENT, [CONTINUOUS], WINDOWS, cells)
    return next(result for result in results if result["branch"] == "treatment-a")


def assert_insufficient_without_an_interval(result, n_reference, n_treatment):
    assert result["state"] == gbstats_compute.INSUFFICIENT_DATA
    assert result["n_reference"] == n_reference
    assert result["n_treatment"] == n_treatment
    assert result.get("error") is None
    for bound in ("point", "lower", "upper", "absolute"):
        assert result.get(bound) is None


def test_a_metric_constant_across_every_unit_is_insufficient_data_not_a_zero_interval():
    # Every unit reports the same value, so there is no variance to form an interval from. gbstats
    # answers with a zero interval rather than failing, which must not read as a measured null.
    reference = branch_moments(800, 1.0, 0.0, 1.0, 1.0, 0.0)
    candidate = branch_moments(600, 1.0, 0.0, 1.0, 1.0, 0.0)

    assert_insufficient_without_an_interval(compute_pair(reference, candidate), 800, 600)


def test_a_reference_branch_of_all_zeros_is_insufficient_data_not_a_zero_interval():
    reference = branch_moments(800, 0.0, 0.0, 1.0, 1.0, 0.0)
    candidate = branch_moments(600, 0.2, 0.5, 1.0, 1.0, 0.3)

    assert_insufficient_without_an_interval(compute_pair(reference, candidate), 800, 600)


def test_a_zero_adjusted_reference_mean_has_no_interval():
    # A theta of one against a covariate mean equal to the metric mean adjusts the reference mean
    # to exactly zero while leaving its variance positive, which is gbstats' zero-baseline case.
    interval = gbstats_compute.sequential_interval(
        branch(1000, 1.0, 2.0, 1.5), branch(1000, 1.1, 2.42, 1.65), 1.0
    )

    assert interval["point"] is None
    assert interval["lower"] is None
    assert interval["upper"] is None
    assert interval["absolute"] == dict(point=None, lower=None, upper=None)


def test_any_other_gbstats_message_is_an_error_carrying_that_message(monkeypatch):
    # No input the job produces reaches gbstats' other stand-in results, so the library's own
    # test object is made to return one.
    monkeypatch.setattr(
        gbstats_compute.SequentialTwoSidedTTest,
        "compute_result",
        lambda test: test._default_output(error_message="SOME_OTHER_GBSTATS_MESSAGE"),
    )

    result = compute_pair(branch(1000, 1.0, 2.0, 1.5), branch(1000, 1.1, 2.42, 1.65))

    assert result["state"] == gbstats_compute.ERROR
    assert "SOME_OTHER_GBSTATS_MESSAGE" in result["error"]
    assert "point" not in result


RATIO = metrics.Ratio(
    name="ad_click_rate",
    numerator=metrics.Metric(
        name="ad_clicks", source="newtab", reducer=None, window_rules=(metrics.CUMULATIVE_WEEKLY,)
    ),
    denominator=metrics.Metric(
        name="search_count",
        source="newtab",
        reducer=None,
        window_rules=(metrics.CUMULATIVE_WEEKLY,),
    ),
    window_rules=(metrics.CUMULATIVE_WEEKLY,),
)


def ratio_branch(records):
    """One branch's ratio sufficient-stats cell, summed from explicit per-unit
    (m_post, d_post, m_pre, d_pre) tuples the same way `sql_generation.ratio_rollup_select` sums
    per-unit window values in BigQuery."""
    n = len(records)
    return dict(
        n=n,
        sum_m=sum(m_post for m_post, d_post, m_pre, d_pre in records),
        sum_m_squares=sum(m_post**2 for m_post, d_post, m_pre, d_pre in records),
        sum_d=sum(d_post for m_post, d_post, m_pre, d_pre in records),
        sum_d_squares=sum(d_post**2 for m_post, d_post, m_pre, d_pre in records),
        sum_m_d=sum(m_post * d_post for m_post, d_post, m_pre, d_pre in records),
        pre_sum_m=sum(m_pre for m_post, d_post, m_pre, d_pre in records),
        pre_sum_m_squares=sum(m_pre**2 for m_post, d_post, m_pre, d_pre in records),
        pre_sum_d=sum(d_pre for m_post, d_post, m_pre, d_pre in records),
        pre_sum_d_squares=sum(d_pre**2 for m_post, d_post, m_pre, d_pre in records),
        pre_sum_m_d=sum(m_pre * d_pre for m_post, d_post, m_pre, d_pre in records),
        sum_m_post_m_pre=sum(m_post * m_pre for m_post, d_post, m_pre, d_pre in records),
        sum_d_post_d_pre=sum(d_post * d_pre for m_post, d_post, m_pre, d_pre in records),
        sum_m_post_d_pre=sum(m_post * d_pre for m_post, d_post, m_pre, d_pre in records),
        sum_m_pre_d_post=sum(m_pre * d_post for m_post, d_post, m_pre, d_pre in records),
    )


REFERENCE_RECORDS = [(1, 10, 1, 10), (2, 11, 2, 9), (0, 9, 1, 11), (3, 12, 2, 10), (1, 8, 0, 8)]
TREATMENT_RECORDS = [(2, 10, 1, 10), (3, 11, 2, 9), (1, 9, 1, 11), (4, 12, 2, 10), (2, 8, 1, 8)]


def test_adjusted_ratio_builds_the_statistic_gbstats_expects():
    cell = dict(
        n=5,
        sum_m=11.0,
        sum_m_squares=31.0,
        sum_d=50.0,
        sum_d_squares=510.0,
        sum_m_d=116.0,
        pre_sum_m=7.0,
        pre_sum_m_squares=11.0,
        pre_sum_d=48.0,
        pre_sum_d_squares=470.0,
        sum_m_post_m_pre=16.0,
        sum_d_post_d_pre=481.0,
        sum_m_post_d_pre=107.0,
        sum_m_pre_d_post=70.0,
        pre_sum_m_d=68.0,
    )

    statistic = gbstats_compute.adjusted_ratio(cell, theta=0.5)

    assert statistic.n == 5
    assert statistic.theta == 0.5
    assert statistic.m_statistic_post.sum == 11.0
    assert statistic.m_statistic_post.sum_squares == 31.0
    assert statistic.d_statistic_post.sum == 50.0
    assert statistic.d_statistic_post.sum_squares == 510.0
    assert statistic.m_statistic_pre.sum == 7.0
    assert statistic.m_statistic_pre.sum_squares == 11.0
    assert statistic.d_statistic_pre.sum == 48.0
    assert statistic.d_statistic_pre.sum_squares == 470.0
    assert statistic.m_post_d_post_sum_of_products == 116.0
    assert statistic.m_pre_d_pre_sum_of_products == 68.0
    assert statistic.m_post_m_pre_sum_of_products == 16.0
    assert statistic.d_post_d_pre_sum_of_products == 481.0
    assert statistic.m_post_d_pre_sum_of_products == 107.0
    assert statistic.m_pre_d_post_sum_of_products == 70.0


def test_pooled_theta_ratio_matches_gbstats_own_two_branch_formula():
    # gbstats fits a ratio's theta from exactly the pair under test
    # (`compute_theta_regression_adjusted_ratio`); with exactly two branches,
    # `pooled_theta_ratio`'s sum-over-every-branch generalisation must return the same value.
    reference = ratio_branch(REFERENCE_RECORDS)
    treatment = ratio_branch(TREATMENT_RECORDS)

    pooled = gbstats_compute.pooled_theta_ratio([reference, treatment])

    reference_statistic = gbstats_compute.adjusted_ratio(reference, theta=1)
    treatment_statistic = gbstats_compute.adjusted_ratio(treatment, theta=1)
    expected = compute_theta_regression_adjusted_ratio(reference_statistic, treatment_statistic)

    assert pooled == pytest.approx(expected)


TREATMENT_B_RECORDS = [(0, 9, 1, 11), (1, 10, 0, 9), (2, 11, 2, 8), (1, 8, 1, 10), (3, 12, 2, 9)]


def test_three_branches_pool_a_different_theta_ratio_than_either_pair_would():
    # The ratio analogue of test_three_branches_pool_a_different_theta_than_either_pair_would:
    # pooling a ratio's theta over three branches is not the same number as pooling over either
    # pair alone, so a three-arm experiment's ratio intervals move rather than being relabelled.
    reference = ratio_branch(REFERENCE_RECORDS)
    treatment_a = ratio_branch(TREATMENT_RECORDS)
    treatment_b = ratio_branch(TREATMENT_B_RECORDS)

    pooled = gbstats_compute.pooled_theta_ratio([reference, treatment_a, treatment_b])

    for candidate in (treatment_a, treatment_b):
        pairwise = gbstats_compute.pooled_theta_ratio([reference, candidate])
        assert pooled != pytest.approx(pairwise, rel=1e-6)


def test_pooled_theta_ratio_is_zero_with_fewer_than_two_units():
    assert gbstats_compute.pooled_theta_ratio([ratio_branch(REFERENCE_RECORDS[:1])]) == 0.0


def test_pooled_theta_ratio_is_zero_when_the_covariate_never_moves():
    # Constant m_pre/d_pre across every unit, the ratio analogue of
    # test_theta_is_zero_when_the_covariate_never_moves: the pre-period has no variance, so the
    # <= 0 guard falls back to no adjustment rather than dividing by (near) zero.
    flat_records = [(1, 10, 1, 1), (2, 11, 1, 1), (0, 9, 1, 1), (3, 12, 1, 1)]

    flat = ratio_branch(flat_records)
    assert gbstats_compute.pooled_theta_ratio([flat, flat]) == 0.0


def test_a_ratio_metric_produces_a_populated_interval_through_compute_statistics():
    cells = {
        ("ad_click_rate", "cumu:1", "all_enrolled", "control"): ratio_branch(REFERENCE_RECORDS),
        ("ad_click_rate", "cumu:1", "all_enrolled", "treatment-a"): ratio_branch(
            TREATMENT_RECORDS
        ),
    }

    results = {
        result["branch"]: result
        for result in gbstats_compute.compute_statistics(EXPERIMENT, [RATIO], WINDOWS, cells)
    }
    treatment = results["treatment-a"]

    assert treatment["state"] in (gbstats_compute.FORMING, gbstats_compute.CONFIDENT)
    assert treatment["lower"] < treatment["point"] < treatment["upper"]
    assert treatment["theta"] == pytest.approx(
        gbstats_compute.pooled_theta_ratio(
            [ratio_branch(REFERENCE_RECORDS), ratio_branch(TREATMENT_RECORDS)]
        )
    )
    assert "absolute" in treatment


def test_a_ratio_cell_below_the_minimum_units_is_insufficient_data():
    cells = {
        ("ad_click_rate", "cumu:1", "all_enrolled", "control"): ratio_branch(REFERENCE_RECORDS),
        ("ad_click_rate", "cumu:1", "all_enrolled", "treatment-a"): ratio_branch(
            TREATMENT_RECORDS[:1]
        ),
    }

    results = {
        result["branch"]: result
        for result in gbstats_compute.compute_statistics(EXPERIMENT, [RATIO], WINDOWS, cells)
    }

    assert results["treatment-a"]["state"] == gbstats_compute.INSUFFICIENT_DATA
    assert results["treatment-a"]["n_treatment"] == 1


def test_a_ratio_gets_only_the_windows_of_the_families_it_declares():
    windows = gbstats_compute.windows_for_metric(RATIO, WINDOWS)

    assert [window.label for window in windows] == ["cumu:1"]


def test_an_ordinary_metric_and_a_ratio_coexist_in_one_compute_statistics_call():
    # The real case once a ratio ships: an experiment's metrics list is a mix of its ordinary
    # guardrails and a ratio, passed to compute_statistics together. Each metric's own pooled_theta
    # function must be used for its own cells, with no cross-talk between the two dispatch paths.
    cells = {
        ("active_hours", "cumu:1", "all_enrolled", "control"): branch(1000, 1.0, 2.0, 1.5),
        ("active_hours", "cumu:1", "all_enrolled", "treatment-a"): branch(1000, 1.1, 2.42, 1.65),
        ("ad_click_rate", "cumu:1", "all_enrolled", "control"): ratio_branch(REFERENCE_RECORDS),
        ("ad_click_rate", "cumu:1", "all_enrolled", "treatment-a"): ratio_branch(
            TREATMENT_RECORDS
        ),
    }

    results = gbstats_compute.compute_statistics(EXPERIMENT, [CONTINUOUS, RATIO], WINDOWS, cells)

    by_metric_branch = {(result["metric"], result["branch"]): result for result in results}
    assert {metric for metric, _ in by_metric_branch} == {"active_hours", "ad_click_rate"}
    mean_treatment = by_metric_branch[("active_hours", "treatment-a")]
    ratio_treatment = by_metric_branch[("ad_click_rate", "treatment-a")]
    assert mean_treatment["state"] in (gbstats_compute.FORMING, gbstats_compute.CONFIDENT)
    assert ratio_treatment["state"] in (gbstats_compute.FORMING, gbstats_compute.CONFIDENT)
    assert mean_treatment["theta"] == pytest.approx(
        gbstats_compute.pooled_theta(
            [branch(1000, 1.0, 2.0, 1.5), branch(1000, 1.1, 2.42, 1.65)]
        )
    )
    assert ratio_treatment["theta"] == pytest.approx(
        gbstats_compute.pooled_theta_ratio(
            [ratio_branch(REFERENCE_RECORDS), ratio_branch(TREATMENT_RECORDS)]
        )
    )
