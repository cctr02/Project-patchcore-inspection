import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--data_path",
        action="store",
        required=True,
        help="Path to the VisA dataset root directory (contains candle/, capsules/, ...).",
    )


@pytest.fixture(scope="session")
def data_path(request):
    return request.config.getoption("--data_path")
