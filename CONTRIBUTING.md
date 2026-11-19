# Contributing

Thank you for your interest in contributing to **Face_Detection_Python**!

## Development Setup

```bash
make setup
source .venv/bin/activate
```

## Running Tests

```bash
make test
```

## Guidelines

1. Keep modules focused and single-purpose inside `src/face_detector/`.
2. Add unit tests under `tests/` for new backends, overlay modes, or CLI flags.
3. Run `pytest` before opening a pull request.
