def pytest_collection_modifyitems(config, items) -> None:  # noqa: ANN001
    """Fail tests marked with 'skipci' without a reason."""
    for item in items:
        marker = item.get_closest_marker("skipci")
        if marker and marker.kwargs.get("reason") is None:
            msg = f"Test {item.nodeid} uses the 'skipci' marker without providing a reason."
            raise ValueError(msg)
