# --- Global plotting style (paper-ready) ---
import logging
from pathlib import Path
import sys
from matplotlib import rcParams

from stellar.data import StellarProfile

rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.2,
    }
)

# Create logger
logger = logging.getLogger("console_logger")
logger.setLevel(logging.DEBUG)

# Create and configure console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Set formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)

if __name__ == "__main__":
    # directory with the generated profiles
    dir = Path("profile_database/")

    save_dir = Path("profile_database/") / "figs"

    max_rank = 30
    fname = fname = f"cat_odd_6.00_max_rank_{str(max_rank)}_niter_400"
    file = dir / Path(fname + ".json")

    logger.info("Loading profile...")

    if not (file.exists() and file.is_file()):
        raise ValueError("File doesn't exist. Generate it.")

    logger.info("File exists, loading stellar profile and plotting...")

    profile = StellarProfile.from_file(filename=fname, path=dir)

    profile.draw(filename=fname, path=save_dir, text=False, show=True)

    logger.info("All done!")
