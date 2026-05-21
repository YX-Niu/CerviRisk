from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.predict.test_request import main


if __name__ == "__main__":
    main()
