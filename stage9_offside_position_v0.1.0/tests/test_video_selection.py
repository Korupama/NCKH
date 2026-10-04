import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage9_offside_position.video_selection import analysis_source_frame


class SourceSelectionTests(unittest.TestCase):
    def test_selected_pixels_win_over_video_id(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[6:9, 17:20] = 255  # A small ball must stay in exactly this place.
        ok, encoded = cv2.imencode('.png', image)
        self.assertTrue(ok)
        for kind in ('DISPLAYED_VIDEO_FRAME', 'INDEXED_FRAME_PREVIEW'):
            request = dict(video_id='a' * 32, estimated_frame=38276, source_kind=kind,
                           image='data:image/png;base64,' + base64.b64encode(encoded).decode())
            with patch('stage9_offside_position.video_selection.read_video_frame') as decode:
                actual, source = analysis_source_frame(Path('.'), request)
                np.testing.assert_array_equal(actual, image)
                decode.assert_not_called()
                self.assertEqual(source, kind)

    def test_missing_snapshot_is_not_silently_replaced_by_another_frame(self):
        with patch('stage9_offside_position.video_selection.read_video_frame') as decode:
            with self.assertRaises(ValueError):
                analysis_source_frame(Path('.'), dict(video_id='a' * 32, source_kind='DISPLAYED_VIDEO_FRAME'))
            decode.assert_not_called()

    def test_legacy_index_request_still_decodes(self):
        with patch('stage9_offside_position.video_selection.read_video_frame', return_value='frame') as decode:
            actual, _ = analysis_source_frame(Path('.'), dict(video_id='a' * 32, estimated_frame=38276))
            self.assertEqual(actual, 'frame')
            decode.assert_called_once_with(Path('.'), 'a' * 32, 38276)


if __name__ == '__main__':
    unittest.main()
