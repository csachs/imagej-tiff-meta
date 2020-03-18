General Information (2020-03-18)
================================

Christoph Gohlke has created the package `roifile <https://github.com/cgohlke/roifile>`_ after discussions in `issue \#1 <https://github.com/csachs/imagej-tiff-meta/issues/1>`_. I would suggest using ``roifile``.

imagej-tiff-meta
================

Experimental support to read and write overlay (ROI) information from ImageJ/Fiji TIFFs.

Monkey patches Christoph Gohlke's ``tifffile.py`` (included).

BSD licensed like ``tifffile.py``.

Experimental and subject to change without notice.


.. code-block:: python
  
  import numpy as np
  from imagej_tiff_meta import TiffFile, TiffWriter

  # open file and print all overlays
  t = TiffFile('input.tif')
  print(t.pages[0].imagej_tags.parsed_overlays)

  # write file and add overlay
  t = TiffWriter('output.tif')

  t.add_roi(np.array([[5, 5], [5, 10], [10, 10], [10, 5]]), t=2)

  t.save(np.zeros((512,512), dtype=np.uint8))
  t.save(np.zeros((512,512), dtype=np.uint8))
  t.save(np.zeros((512,512), dtype=np.uint8))

  t.close()



