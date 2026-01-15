"""Pytest configuration and fixtures for Wavira tests."""

import pytest


def pytest_addoption(parser):
    """Add command line options for tests."""
    parser.addoption(
        "--mqtt-broker",
        action="store_true",
        default=False,
        help="Run tests that require a live MQTT broker",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and options."""
    # Skip MQTT broker tests unless --mqtt-broker flag is passed
    if not config.getoption("--mqtt-broker"):
        skip_mqtt = pytest.mark.skip(reason="need --mqtt-broker option to run")
        for item in items:
            if "mqtt" in item.keywords:
                item.add_marker(skip_mqtt)
