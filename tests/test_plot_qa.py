"""Check the portable layout gate with synthetic plots."""
import sys
import unittest
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

class PlotQATests(unittest.TestCase):
    def test_equal_panels(self):
        fig, axes = plt.subplots(1, 2)
        for label, ax in zip('ab', axes):
            ax.text(0, 1, label, transform=ax.transAxes)
        report = require_matplotlib_panel_alignment(fig, require_panel_labels=True)
        self.assertEqual(report['failures'], [])
        plt.close(fig)

    def test_misaligned_panels_fail(self):
        fig, axes = plt.subplots(1, 2)
        axes[1].set_position([.6, .2, .1, .4])
        with self.assertRaises(ValueError):
            require_matplotlib_panel_alignment(fig)
        plt.close(fig)
