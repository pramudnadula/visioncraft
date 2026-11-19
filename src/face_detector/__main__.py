"""Allow running `python -m face_detector`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
