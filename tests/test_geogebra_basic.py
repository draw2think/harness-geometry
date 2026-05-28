"""
Basic tests for GeoGebra API integration.

Run with: pytest tests/test_geogebra_basic.py -v
"""

import pytest
from pathlib import Path
from symbolic.integrations.geogebra_api import GeoGebraAPI


@pytest.fixture
def ggb():
    """Create GeoGebra instance for testing."""
    api = GeoGebraAPI(mode="selenium", headless=True)
    api.initialize()
    yield api
    api.cleanup()


def test_basic_point_creation(ggb):
    """Test creating a simple point."""
    result = ggb.eval_command("A = (1, 2)")
    assert result.success, f"Failed to create point: {result.error_message}"

    # Verify point exists
    assert ggb.exists("A"), "Point A should exist"

    # Check coordinates
    coords = ggb.get_coords("A")
    assert coords is not None, "Should get coordinates"
    assert coords == (1, 2), f"Expected (1, 2), got {coords}"


def test_create_triangle(ggb):
    """Test creating a triangle."""
    # Create three points
    ggb.eval_command("A = (0, 0)")
    ggb.eval_command("B = (3, 0)")
    ggb.eval_command("C = (0, 4)")

    # Create triangle
    success, labels = ggb.eval_command_get_labels("Polygon(A, B, C)")
    assert success, "Should create polygon"

    # Check all points exist
    assert ggb.exists("A")
    assert ggb.exists("B")
    assert ggb.exists("C")


def test_get_all_objects(ggb):
    """Test getting all objects."""
    # Create some objects
    ggb.eval_command("A = (0, 0)")
    ggb.eval_command("B = (1, 1)")

    # Get all points
    points = ggb.get_all_object_names("point")
    assert "A" in points
    assert "B" in points


def test_export_png(ggb):
    """Test exporting to PNG."""
    # Create a simple figure
    ggb.eval_command("A = (0, 0)")
    ggb.eval_command("B = (2, 0)")
    ggb.eval_command("C = (1, 2)")
    ggb.eval_command("Polygon(A, B, C)")

    # Export
    output_path = Path("temp/test_triangle.png")
    success = ggb.export_png(output_path)

    assert success, "Should export PNG"
    assert output_path.exists(), "PNG file should exist"
    assert output_path.stat().st_size > 0, "PNG should not be empty"


def test_construction_state(ggb):
    """Test getting construction state."""
    # Create points and line
    ggb.eval_command("A = (0, 0)")
    ggb.eval_command("B = (3, 4)")
    ggb.eval_command("Line(A, B)")

    # Get state
    state = ggb.get_construction_state()

    assert len(state["points"]) >= 2, "Should have at least 2 points"
    assert "A" in state["objects"]
    assert "B" in state["objects"]


def test_reset(ggb):
    """Test resetting construction."""
    # Create objects
    ggb.eval_command("A = (1, 1)")
    ggb.eval_command("B = (2, 2)")

    assert ggb.exists("A")

    # Reset
    ggb.reset()

    # Verify cleared
    assert not ggb.exists("A"), "Objects should be cleared after reset"


@pytest.mark.skip(reason="Requires working implementation")
def test_isosceles_triangle_proof(ggb):
    """
    Test proof: If AB = AC, then angle B = angle C.

    This is a simple proof that should work once the full system is implemented.
    """
    # Set up isosceles triangle
    ggb.eval_command("A = (0, 2)")
    ggb.eval_command("B = (-1, 0)")
    ggb.eval_command("C = (1, 0)")

    # Verify AB = AC
    ab = ggb.get_value("Distance(A, B)")
    ac = ggb.get_value("Distance(A, C)")
    assert abs(ab - ac) < 0.001, "Triangle should be isosceles"

    # Get angles
    angle_b = ggb.get_value("Angle(C, B, A)")
    angle_c = ggb.get_value("Angle(B, C, A)")

    # Verify angles are equal
    assert abs(angle_b - angle_c) < 0.001, "Angles B and C should be equal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
