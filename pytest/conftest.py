import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="Enable plotting in reliability tests"
    )

@pytest.fixture
def plot_enabled(request):
    return request.config.getoption("--plot")
