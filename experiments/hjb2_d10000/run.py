"""One-command hjb2_d10000 paper reproduction."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dfm_repro.cli import main
if __name__=="__main__":main("hjb2_d10000")
