"""Cloud stats updater (run by GitHub Actions every morning). Downloads every finished game
from ESPN into data/ - the phone app reads it from there."""
import sys
from datetime import date, timedelta
from pathlib import Path

from q1model import espn_data

data_dir = Path(__file__).parent / "data"
start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
print(espn_data.update(data_dir, start=start, end=date.today() - timedelta(days=1)))
