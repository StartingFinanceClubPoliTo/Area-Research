from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rough_processes as rp  # noqa: E402
import solve_rough_hjb as hjb  # noqa: E402
from generate_article_figures import FigureConfig  # noqa: E402
from interactive_fbm_dashboard import FBMDashboard  # noqa: E402


class RoughProcessTests(unittest.TestCase):
    def test_davies_harte_is_deterministic_and_matches_fbm_covariance(self) -> None:
        model = rp.DaviesHarteFractionalBrownianMotion(
            rp.SimulationGrid(24, 0.75), 0.3
        )
        first = model.simulate(4000, 17)
        second = model.simulate(4000, 17)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, 0], np.zeros(first.shape[0]))
        _, rmse = rp.covariance_error(first, model.covariance)
        self.assertLess(rmse, 0.025)

    def test_lagwise_covariance_error_uses_constant_lag_diagonals(self) -> None:
        difference = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, -3.0, 4.0],
                [2.0, 4.0, 5.0],
            ]
        )
        np.testing.assert_allclose(
            rp.lagwise_covariance_error(difference),
            np.array([8.0 / 3.0, 2.5, 2.0]),
        )
        np.testing.assert_allclose(
            rp.lagwise_covariance_error(difference, reduction="rmse"),
            np.array([np.sqrt(34.0 / 3.0), np.sqrt(8.5), 2.0]),
        )

        model = rp.DaviesHarteFractionalBrownianMotion(rp.SimulationGrid(30), 0.2)
        paths = model.simulate(20, 31)
        covariance_difference, _ = rp.covariance_error(paths, model.covariance)
        self.assertEqual(rp.lagwise_covariance_error(covariance_difference)[-1], 0.0)

        with self.assertRaises(ValueError):
            rp.lagwise_covariance_error(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            rp.lagwise_covariance_error(np.eye(3), reduction="unknown")

    def test_article_figure_defaults_match_reference_panels(self) -> None:
        config = FigureConfig()
        self.assertEqual(config.sample_paths, 10)
        self.assertEqual(config.lift_paths, 10)
        self.assertEqual(config.lift_factors, 10)

    def test_functional_api_matches_class_api(self) -> None:
        grid = rp.SimulationGrid(24, 0.75)

        fbm_model = rp.FractionalBrownianMotion(grid, 0.3)
        expected_fbm = fbm_model.simulate(5, 7)
        actual_fbm, time, covariance = rp.simulate_fbm_cholesky(24, 0.3, 5, 0.75, 7)
        np.testing.assert_array_equal(actual_fbm, expected_fbm)
        np.testing.assert_array_equal(time, grid.time)
        np.testing.assert_array_equal(covariance, fbm_model.covariance)

        volterra_model = rp.VolterraProcess(grid, 0.3)
        expected_volterra, expected_brownian = volterra_model.simulate(5, 11)
        actual_volterra, actual_brownian, _, kernel = rp.simulate_volterra(
            24, 0.3, 5, 0.75, 11
        )
        np.testing.assert_array_equal(actual_volterra, expected_volterra)
        np.testing.assert_array_equal(actual_brownian, expected_brownian)
        np.testing.assert_array_equal(kernel, volterra_model.kernel)

        lift_model = rp.MarkovianLift(grid, 0.3, 4)
        expected_lift = lift_model.simulate(5, 13)
        actual_lift, _, rates, weights = rp.simulate_markov_lift(
            24, 0.3, 5, 4, 0.75, 13
        )
        np.testing.assert_allclose(actual_lift, expected_lift, rtol=0.0, atol=1e-14)
        np.testing.assert_array_equal(rates, lift_model.rates)
        np.testing.assert_array_equal(weights, lift_model.weights)

    def test_reference_outputs_remain_numerically_stable(self) -> None:
        lift, _, _, _ = rp.simulate_markov_lift(24, 0.3, 5, 4, 0.75, 13)
        self.assertAlmostEqual(float(lift.sum()), 20.417716929773718, places=12)
        self.assertAlmostEqual(float(lift.std()), 0.8933035942231591, places=12)

        _, _, _, kernel = rp.simulate_volterra(24, 0.3, 5, 0.75, 11)
        self.assertTrue(np.allclose(np.triu(kernel), 0.0))

    def test_validation_rejects_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            rp.SimulationGrid(2)
        with self.assertRaises(ValueError):
            rp.FractionalBrownianMotion(rp.SimulationGrid(10), 1.0)
        with self.assertRaises(ValueError):
            rp.MarkovianLift(rp.SimulationGrid(10), factors=0)


class HJBTests(unittest.TestCase):
    @staticmethod
    def params() -> hjb.ModelParams:
        return hjb.ModelParams(
            n_steps=30,
            trading_minutes=30,
            q_max=2,
            lift_dim=3,
            y_points=31,
            mc_paths=12,
            representative_seed=17,
            mc_seed=19,
            hurst=0.3,
            hurst_values=(0.3,),
        )

    def test_hjb_surface_regression(self) -> None:
        params = self.params()
        lift = hjb.build_lift(params)
        solution = hjb.solve_hjb(params, lift)
        self.assertEqual(solution["theta"].shape, (31, 5, 31))
        self.assertAlmostEqual(float(solution["theta"].sum()), 12598.663226075154, places=10)
        self.assertAlmostEqual(float(solution["delta_bid"].sum()), 7886.785734444949, places=10)
        self.assertAlmostEqual(float(solution["delta_ask"].sum()), 7886.785734444949, places=10)

    def test_market_environment_is_shared_by_both_policies(self) -> None:
        params = self.params()
        lift = hjb.build_lift(params)
        solution = hjb.solve_hjb(params, lift)
        shocks = hjb.SimulationShocks.draw(params, np.random.default_rng(123))
        market = hjb.build_market_environment(params, lift, shocks)
        simulator = hjb.PolicySimulator(params, lift, solution)

        naive = simulator.simulate("naive", shocks, market)
        optimal = simulator.simulate("hjb", shocks, market)
        np.testing.assert_array_equal(naive["s"], optimal["s"])
        np.testing.assert_array_equal(naive["y"], optimal["y"])
        np.testing.assert_array_equal(naive["vol"], optimal["vol"])

        implicit = hjb.simulate_policy(params, lift, solution, "naive", shocks)
        np.testing.assert_allclose(implicit["s"], naive["s"], rtol=0.0, atol=1e-14)
        self.assertAlmostEqual(float(implicit["wealth"][-1]), float(naive["wealth"][-1]))

        with self.assertRaises(ValueError):
            simulator.simulate("unknown", shocks, market)

    def test_experiment_class_preserves_functional_entry_point(self) -> None:
        params = self.params()
        class_result = hjb.RoughHJBExperiment(params).run()
        function_result = hjb.run_case(params)
        self.assertEqual(class_result["chosen_seed"], function_result["chosen_seed"])
        for mode in ("naive", "hjb"):
            self.assertAlmostEqual(
                class_result["mc"]["stats"][mode]["mean_pnl"],
                function_result["mc"]["stats"][mode]["mean_pnl"],
            )


class DashboardTests(unittest.TestCase):
    def test_redraw_does_not_accumulate_colorbar_axes(self) -> None:
        dashboard = FBMDashboard()
        try:
            dashboard._redraw()
            axes_after_first_draw = len(dashboard.figure.axes)
            dashboard._redraw()
            self.assertEqual(len(dashboard.figure.axes), axes_after_first_draw)
            self.assertEqual(len(dashboard._colorbars), 2)
        finally:
            plt.close(dashboard.figure)


class StandaloneNotebookTests(unittest.TestCase):
    def test_notebook_is_self_contained_and_generates_all_article_images(self) -> None:
        notebook_path = ROOT / "src" / "Rough_Volatility_HJB_Standalone.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        local_modules = (
            "rough_processes",
            "dashboard_base",
            "generate_article_figures",
            "solve_rough_hjb",
            "interactive_fbm_dashboard",
            "interactive_volterra_dashboard",
            "interactive_lift_dashboard",
        )
        for module in local_modules:
            self.assertIsNone(
                re.search(rf"(^|\n)\s*(?:from|import)\s+{module}\b", code),
                msg=f"Notebook imports local module {module}",
            )

        self.assertIn("class ArticleFigureGenerator", code)
        self.assertIn("class RoughHJBExperiment", code)
        self.assertIn("def generate_all_article_images", code)
        for image_name in (
            "img_4.png",
            "img_5.png",
            "hjb_surfaces.png",
            "hjb_simulation.png",
        ):
            self.assertIn(image_name, code)


if __name__ == "__main__":
    unittest.main()
