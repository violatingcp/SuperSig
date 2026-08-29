"""Exp 70's GCD arms: the SimCLR term sees every image, the SupCon term only
the labelled ones (label -1 = held out), and the arms are opt-in."""
import importlib, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))


def test_gcd_step_masks_unlabelled_from_supcon_and_arms_are_opt_in():
    exp70 = importlib.import_module("70_cars_ft_suite")
    from supersig.config import DEVICE
    torch.manual_seed(0)
    model = torch.nn.Linear(6, 4).to(DEVICE)
    v1, v2 = torch.randn(8, 6), torch.randn(8, 6)
    y = torch.tensor([0, 0, 1, 1, -1, -1, -1, -1])
    step = exp70.make_gcd_step(0.0, 8)
    con, reg = step(model, v1, v2, y)
    assert torch.isfinite(con) and float(reg) == 0.0
    # all-unlabelled batch: SupCon term drops out, SimCLR term remains finite
    con2, _ = step(model, v1, v2, torch.full((8,), -1))
    assert torch.isfinite(con2)
    step5 = exp70.make_gcd_step(5.0, 8)
    _, reg5 = step5(model, v1, v2, y)
    assert float(reg5) > 0
    assert exp70.GCD_ARMS == {"gcd-ft", "gcd-sigreg-ft"}
    assert set(exp70.COLORS) >= exp70.GCD_ARMS


def test_masked_loader_hides_holdout_labels():
    exp70 = importlib.import_module("70_cars_ft_suite")
    class Toy:
        def __len__(self): return 4
        def __getitem__(self, i): return torch.zeros(2), i
    ds = exp70.MaskedLabelSubset(Toy(), [0, 1, 2, 3], [0, -1, 2, -1])
    assert [ds[i][1] for i in range(4)] == [0, -1, 2, -1]
