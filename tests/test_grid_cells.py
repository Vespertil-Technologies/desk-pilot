from computer import GRID_COLS, GRID_ROWS, compute_grid_cells


def test_default_grid_has_expected_cell_count():
    cells = compute_grid_cells(1920, 1080, GRID_COLS, GRID_ROWS)
    assert len(cells) == GRID_COLS * GRID_ROWS


def test_grid_cell_names_are_letter_then_number():
    cells = compute_grid_cells(800, 600, 4, 4)
    assert set(cells.keys()) == {f"{c}{r}" for c in "ABCD" for r in range(1, 5)}


def test_top_left_cell_centred_correctly():
    cells = compute_grid_cells(800, 600, 4, 4)
    # 800/4=200 wide, 600/4=150 tall, A1 center is (100, 75).
    assert cells["A1"].x == 100
    assert cells["A1"].y == 75


def test_bottom_right_cell_centred_correctly():
    cells = compute_grid_cells(800, 600, 4, 4)
    # D4 center is (700, 525) for the same grid.
    assert cells["D4"].x == 700
    assert cells["D4"].y == 525


def test_all_cells_within_image_bounds():
    width, height = 1920, 1080
    cells = compute_grid_cells(width, height, GRID_COLS, GRID_ROWS)
    for cell in cells.values():
        assert 0 <= cell.x < width
        assert 0 <= cell.y < height


def test_cells_are_grid_aligned():
    cells = compute_grid_cells(1000, 500, 5, 5)
    # Column A's centre x is the same in every row.
    a_xs = {cells[f"A{r}"].x for r in range(1, 6)}
    assert len(a_xs) == 1
    # Row 1's centre y is the same across columns.
    row1_ys = {cells[f"{c}1"].y for c in "ABCDE"}
    assert len(row1_ys) == 1
