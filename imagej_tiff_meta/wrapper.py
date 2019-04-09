# encoding: utf-8
# Copyright (c) 2017, Christian C. Sachs
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # code tend to throw warnings because of missing C extensions
    import imagej_tiff_meta.tifffile as patchy_tifffile

# https://github.com/imagej/imagej1/blob/2a6c191b027b5b1f5f22484506159f80adda21c5/ij/io/TiffDecoder.java

CONSTANT_MAGIC_NUMBER = 0x494a494a  # "IJIJ"
CONSTANT_INFO = 0x696e666f  # "info" (Info image property)
CONSTANT_LABELS = 0x6c61626c  # "labl" (slice labels)
CONSTANT_RANGES = 0x72616e67  # "rang" (display ranges)
CONSTANT_LUTS = 0x6c757473  # "luts" (channel LUTs)
CONSTANT_ROI = 0x726f6920  # "roi " (ROI)
CONSTANT_OVERLAY = 0x6f766572  # "over" (overlay)

CONST_IJ_POLYGON,\
    CONST_IJ_RECT,\
    CONST_IJ_OVAL,\
    CONST_IJ_LINE,\
    CONST_IJ_FREELINE,\
    CONST_IJ_POLYLINE,\
    CONST_IJ_NOROI,\
    CONST_IJ_FREEHAND,\
    CONST_IJ_TRACED,\
    CONST_IJ_ANGLE,\
    CONST_IJ_POINT\
    = range(11)

# https://docs.oracle.com/javase/7/docs/api/constant-values.html#java.awt.geom.PathIterator

CONST_PATH_ITERATOR_SEG_MOVETO, \
    CONST_PATH_ITERATOR_SEG_LINETO, \
    CONST_PATH_ITERATOR_SEG_QUADTO, \
    CONST_PATH_ITERATOR_SEG_CUBICTO, \
    CONST_PATH_ITERATOR_SEG_CLOSE = range(5)

# https://github.com/imagej/imagej1/blob/86280b4e0756d1f4c0fcb44ac7410138e8e6a6d8/ij/io/RoiDecoder.java

CONST_IJ_OPT_SPLINE_FIT,\
    CONST_IJ_OPT_DOUBLE_HEADED, \
    CONST_IJ_OPT_OUTLINE, \
    CONST_IJ_OPT_OVERLAY_LABELS,\
    CONST_IJ_OPT_OVERLAY_NAMES,\
    CONST_IJ_OPT_OVERLAY_BACKGROUNDS,\
    CONST_IJ_OPT_OVERLAY_BOLD, \
    CONST_IJ_OPT_SUB_PIXEL_RESOLUTION,\
    CONST_IJ_OPT_DRAW_OFFSET,\
    CONST_IJ_OPT_ZERO_TRANSPARENT,\
    = [2**bits for bits in range(10)]

IMAGEJ_ROI_HEADER_BEGIN = [
    ('_iout', '4a1'),  # always b'Iout'
    ('version', 'i2'),
    ('roi_type', 'i1'),
    ('_pad_byte', 'u1'),
    ('top', 'i2'),
    ('left', 'i2'),
    ('bottom', 'i2'),
    ('right', 'i2'),
    ('n_coordinates', 'i2')]

IMAGEJ_ROI_HEADER_PIXEL_RESOLUTION_MIDDLE = [
    ('x1', 'i4'),
    ('y1', 'i4'),
    ('x2', 'i4'),
    ('y2', 'i4'),
]

IMAGEJ_ROI_HEADER_SUB_PIXEL_RESOLUTION_MIDDLE = [
    ('x1', 'f4'),
    ('y1', 'f4'),
    ('x2', 'f4'),
    ('y2', 'f4'),
]

IMAGEJ_ROI_HEADER_END = [
    ('stroke_width', 'i2'),
    ('shape_roi_size', 'i4'),
    ('stroke_color', 'i4'),
    ('fill_color', 'i4'),
    ('subtype', 'i2'),
    ('options', 'i2'),
    ('arrow_style_or_aspect_ratio', 'u1'),
    ('arrow_head_size', 'u1'),
    ('rounded_rect_arc_size', 'i2'),
    ('position', 'i4'),
    ('header2_offset', 'i4'),
]

IMAGEJ_ROI_HEADER = IMAGEJ_ROI_HEADER_BEGIN +\
                    IMAGEJ_ROI_HEADER_PIXEL_RESOLUTION_MIDDLE +\
                    IMAGEJ_ROI_HEADER_END

IMAGEJ_ROI_HEADER_SUB_PIXEL = IMAGEJ_ROI_HEADER_BEGIN +\
                              IMAGEJ_ROI_HEADER_SUB_PIXEL_RESOLUTION_MIDDLE +\
                              IMAGEJ_ROI_HEADER_END

IMAGEJ_ROI_HEADER2 = [
    ('_nil', 'i4'),
    ('c', 'i4'),
    ('z', 'i4'),
    ('t', 'i4'),
    ('name_offset', 'i4'),
    ('name_length', 'i4'),
    #
    ('overlay_label_color', 'i4'),
    ('overlay_font_size', 'i2'),
    ('available_byte1', 'i1'),
    ('image_opacity', 'i1'),
    ('image_size', 'i4'),
    ('float_stroke_width', 'f4'),
    ('roi_props_offset', 'i4'),
    ('roi_props_length', 'i4'),
    ('counters_offset', 'i4')
]

IMAGEJ_META_HEADER = [
    ('magic', 'i4'),
    ('type', 'i4'),
    ('count', 'i4'),
]

IJM_ROI_VERSION = 226


def new_record(dtype, data=None, offset=0):
    tmp = np.recarray(shape=(1,), dtype=dtype, aligned=False, buf=data, offset=offset).newbyteorder('>')[0]
    if data is None:
        tmp.fill(0)  # recarray does not initialize empty memory! that's pretty scary
    return tmp


IMAGEJ_SUPPORTED_OVERLAYS = {
        CONST_IJ_POLYGON,
        CONST_IJ_FREEHAND,
        CONST_IJ_TRACED,
        CONST_IJ_POLYLINE,
        CONST_IJ_FREELINE,
        CONST_IJ_ANGLE,
        CONST_IJ_POINT
    }


def shape_array_to_coordinates(shape_array):
    results = []
    result = []
    n = 0

    last_moveto = 0

    while n < len(shape_array):
        op = int(shape_array[n])
        if op == CONST_PATH_ITERATOR_SEG_MOVETO:
            if n > 0:
                results.append(np.array(result))
                result = []
            result.append([shape_array[n + 1], shape_array[n + 2]])
            last_moveto = len(result)
            n += 3
        elif op == CONST_PATH_ITERATOR_SEG_LINETO:
            result.append([shape_array[n + 1], shape_array[n + 2]])
            n += 3
        elif op == CONST_PATH_ITERATOR_SEG_CLOSE:
            result.append(result[last_moveto])
            n += 1
        elif op == CONST_PATH_ITERATOR_SEG_QUADTO or op == CONST_PATH_ITERATOR_SEG_CUBICTO:
            raise RuntimeError("Unsupported PathIterator commands in ShapeRoi")

    results.append(np.array(result))
    return results


def imagej_parse_overlay(data):
    header = new_record(IMAGEJ_ROI_HEADER, data=data)
    headerf = new_record(IMAGEJ_ROI_HEADER_SUB_PIXEL, data=data)

    header2 = new_record(IMAGEJ_ROI_HEADER2, data=data, offset=header.header2_offset)

    if header2.name_offset > 0:
        name = str(data[header2.name_offset:header2.name_offset + header2.name_length * 2], 'utf-16be')
    else:
        name = ''

    sub_pixel_resolution = (header.options & CONST_IJ_OPT_SUB_PIXEL_RESOLUTION) and header.version >= 222
    draw_offset = sub_pixel_resolution and (header.options & CONST_IJ_OPT_DRAW_OFFSET)

    sub_pixel_resolution = False

    if sub_pixel_resolution:
        header = headerf

    overlay = dict(
        name=name,
        coordinates=None,
        sub_pixel_resolution=sub_pixel_resolution,
        draw_offset=draw_offset,
    )

    if header.roi_type in IMAGEJ_SUPPORTED_OVERLAYS:
        dtype_to_fetch = np.dtype(np.float32) if sub_pixel_resolution else np.dtype(np.int16)

        coordinates_to_fetch = header.n_coordinates

        if sub_pixel_resolution:
            coordinate_offset = coordinates_to_fetch * np.dtype(np.uint16).itemsize * 2
        else:
            coordinate_offset = 0

        overlay['coordinates'] = np.ndarray(
            shape=(coordinates_to_fetch, 2),
            dtype=dtype_to_fetch.newbyteorder('>'),
            buffer=data[
                   header.itemsize + coordinate_offset:
                   header.itemsize + coordinate_offset + 2 * dtype_to_fetch.itemsize * coordinates_to_fetch
                   ],
            order='F'
        ).copy()

        overlay['multi_coordinates'] = [overlay['coordinates'].copy()]

    elif header.roi_type == CONST_IJ_RECT and header.shape_roi_size > 0:
        # composite / shape ROI ... not pretty to parse
        shape_array = np.ndarray(
            shape=header.shape_roi_size,
            dtype=np.dtype(np.float32).newbyteorder('>'),
            buffer=data[
                   header.itemsize:
                   header.itemsize + np.dtype(np.float32).itemsize * header.shape_roi_size
                   ]
        ).copy()

        overlay['multi_coordinates'] = shape_array_to_coordinates(shape_array)

        for coords in overlay['multi_coordinates']:
            coords -= [header.left, header.top]

        overlay['coordinates'] = next(
            iter(
                sorted(
                    overlay['multi_coordinates'],
                    key=lambda coords: len(coords),
                    reverse=True
                )
            )
        )

    for to_insert in [header, header2]:
        for key in to_insert.dtype.names:
            if key[0] == '_':
                continue
            overlay[key] = np.asscalar(getattr(to_insert, key))

    return overlay


def imagej_create_roi(points, name=None, c=-1, z=-1, t=-1, position=-1, index=None):
    if name is None:
        if index is None:
            name = 'F%02d-%x' % (t+1, np.random.randint(0, 2**32 - 1),)
        else:
            name = 'F%02d-C%d' % (t+1, index,)

    points = points.copy()
    left, top = points[:, 0].min(), points[:, 1].min()
    points[:, 0] -= left
    points[:, 1] -= top

    sub_pixel_resolution = False

    if points.dtype == np.float32 or points.dtype == np.float64:
        sub_pixel_resolution = True

    encoded_data = points.astype(np.dtype(np.int16).newbyteorder('>')).tobytes(order='F')

    encoded_data_size = len(encoded_data)

    if sub_pixel_resolution:
        points[:, 0] += left
        points[:, 1] += top
        sub_pixel_data = points.astype(np.dtype(np.float32).newbyteorder('>')).tobytes(order='F')
        encoded_data += sub_pixel_data
        encoded_data_size += len(sub_pixel_data)

    header = new_record(IMAGEJ_ROI_HEADER) if not sub_pixel_resolution else new_record(IMAGEJ_ROI_HEADER_SUB_PIXEL)

    header._iout = b'I', b'o', b'u', b't'

    header.version = IJM_ROI_VERSION

    header.roi_type = CONST_IJ_FREEHAND  # CONST_IJ_POLYGON

    header.left = left
    header.top = top

    header.n_coordinates = len(points)

    header.options = 40

    if sub_pixel_resolution:
        header.options |= CONST_IJ_OPT_SUB_PIXEL_RESOLUTION

    header.position = position + 1
    header.header2_offset = header.itemsize + encoded_data_size

    header2 = new_record(IMAGEJ_ROI_HEADER2)

    # either: position if only time series
    # or position 0 and c / z / t if hyperstack

    header2.c = c + 1
    header2.z = z + 1
    header2.t = t + 1

    header2.name_offset = header.header2_offset + header2.itemsize
    header2.name_length = len(name)

    return header.tobytes() + encoded_data + header2.tobytes() + name.encode('utf-16be')


def imagej_prepare_metadata(overlays):
    mh = new_record(IMAGEJ_META_HEADER)

    mh.magic = CONSTANT_MAGIC_NUMBER

    # mh.type = CONSTANT_ROI
    mh.type = CONSTANT_OVERLAY
    mh.count = len(overlays)

    meta_data = mh.tobytes() + b''.join(overlays)

    byte_counts = [mh.itemsize] + [len(r) for r in overlays]  # len of overlays

    return meta_data, byte_counts

###
# Monkey patching
###


def TiffWriter___init__(self, filename):
    self.__original_init__(
        filename,
        bigtiff=False,
        imagej=True,
        byteorder='>'
    )

    self._ijm_roi_data = []
    self._ijm_rois_per_frame = {}
    self._ijm_first_written = False


def TiffWriter_add_roi(self, points, name=None, c=-1, z=-1, t=-1, position=-1):
    index = None
    if name is None:
        if t not in self._ijm_rois_per_frame:
            self._ijm_rois_per_frame[t] = 0
        self._ijm_rois_per_frame[t] += 1
        index = self._ijm_rois_per_frame[t]

    self._ijm_roi_data.append(imagej_create_roi(points, name=name, c=c, z=z, t=t, position=position, index=index))


def TiffWriter_new_save(self, data, **kwargs):
    if self._ijm_first_written or len(self._ijm_roi_data) == 0:
        return self.__original_save(data, **kwargs)

    meta_data, byte_counts = imagej_prepare_metadata(self._ijm_roi_data)

    self._ijm_first_written = True

    extratags = [
        (50838, 'I', len(byte_counts), byte_counts, True),  # byte counts
        (50839, 'B', len(meta_data), np.frombuffer(meta_data, dtype=np.uint8), True),  # meta data
        # (34122, 'I', 1, [self._ijm_frames], True)  # meta data
    ]

    kwargs['extratags'] = kwargs.get('extratags', []) + extratags

    return self.__original_save(data, **kwargs)


def new_imagej_metadata(*args):
    result = __original_imagej_metadata(*args)
    try:
        if 'overlays' in result:
            result['parsed_overlays'] = [
                patchy_tifffile.Record(imagej_parse_overlay(data))
                for data in result['overlays']
            ]
    except Exception as e:
        print(e)

        import traceback
        traceback.print_exc()

    return result


# monkey patching

patchy_tifffile.TiffWriter.__original_init__ = patchy_tifffile.TiffWriter.__init__
patchy_tifffile.TiffWriter.__init__ = TiffWriter___init__
patchy_tifffile.TiffWriter.__original_save = patchy_tifffile.TiffWriter.save
patchy_tifffile.TiffWriter.save = TiffWriter_new_save
patchy_tifffile.TiffWriter.add_roi = TiffWriter_add_roi

TiffWriter = patchy_tifffile.TiffWriter

__original_imagej_metadata = patchy_tifffile.imagej_metadata

patchy_tifffile.imagej_metadata = new_imagej_metadata

TiffFile = patchy_tifffile.TiffFile
