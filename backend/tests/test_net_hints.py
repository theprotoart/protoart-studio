import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import net


class MissingDownloadTests(unittest.TestCase):
    def test_transformers_processor_not_downloaded(self):
        e = OSError("Can't load image processor for 'PekingU/rtdetr_r50vd_coco_o365'. If you were trying to load it "
                    "from 'https://huggingface.co/models', make sure you don't have a local directory with the same name.")
        self.assertTrue(net.looks_like_missing_download(e))

    def test_other_errors_are_not_downloads(self):
        self.assertFalse(net.looks_like_missing_download(ValueError("width must be a multiple of 8")))


if __name__ == "__main__":
    unittest.main()
