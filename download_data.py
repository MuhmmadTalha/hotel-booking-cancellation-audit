"""Download the Hotel Booking Demand dataset into this project.

Run once from the project root:
    python download_data.py
"""

from pathlib import Path
import shutil

import kagglehub


DATASET = "jessemostipak/hotel-booking-demand"
PROJECT_ROOT = Path(__file__).resolve().parent
DESTINATION = PROJECT_ROOT / "data" / "raw" / "hotel_bookings.csv"


def main() -> None:
    source_dir = Path(kagglehub.dataset_download(DATASET))
    source_file = source_dir / "hotel_bookings.csv"
    if not source_file.is_file():
        available = ", ".join(path.name for path in source_dir.iterdir())
        raise FileNotFoundError(
            f"hotel_bookings.csv was not found in {source_dir}. Available: {available}"
        )

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, DESTINATION)
    print(f"Dataset saved to: {DESTINATION}")


if __name__ == "__main__":
    main()
