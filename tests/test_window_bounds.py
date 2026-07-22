"""
window_bounds() must return physical pixels.

The grid the agent clicks through is built from a PyAutoGUI screenshot, which
is always physical pixels. The browser reports its rect in CSS pixels. Mixing
the two made the screenshot-mode clamp reject cells that were well inside the
window on any scaled display.
"""

from computer import BrowserSession, compute_grid_cells


class FakePage:
    """Stands in for a Playwright page, reporting CSS-pixel geometry."""

    def __init__(self, dpr, screen_x=0, screen_y=0, outer_w=1536, outer_h=816):
        self.dpr = dpr
        self.values = [screen_x, screen_y, outer_w, outer_h]

    def evaluate(self, _js):
        return [v * self.dpr for v in self.values]


def _session(page):
    session = BrowserSession()
    session._page = page
    return session


def test_unscaled_display_is_unchanged():
    assert _session(FakePage(dpr=1)).window_bounds() == (0, 0, 1536, 816)


def test_bounds_are_scaled_to_physical_pixels():
    # 125% scaling: a 1536x816 CSS window is 1920x1020 real pixels.
    assert _session(FakePage(dpr=1.25)).window_bounds() == (0, 0, 1920, 1020)


def test_offset_window_origin_is_scaled_too():
    page = FakePage(dpr=2, screen_x=100, screen_y=50, outer_w=800, outer_h=600)
    assert _session(page).window_bounds() == (200, 100, 1600, 1200)


def test_missing_page_returns_none():
    assert BrowserSession().window_bounds() is None


def test_evaluate_failure_returns_none():
    class Broken:
        def evaluate(self, _js):
            raise RuntimeError("page closed")

    assert _session(Broken()).window_bounds() is None


def test_scaled_window_accepts_the_cells_it_covers():
    """A maximized window at 125% scaling should not reject most of its page."""
    screen_w, screen_h = 1920, 1080
    cells = compute_grid_cells(screen_w, screen_h, 20, 15)
    bx, by, bw, bh = _session(FakePage(dpr=1.25)).window_bounds()

    inside = [c for c in cells.values() if bx <= c.x <= bx + bw and by <= c.y <= by + bh]
    # Before the fix this was 176/300, because the bounds came back as 1536x816
    # and silently cut off the right and bottom of the window.
    assert len(inside) >= 280
