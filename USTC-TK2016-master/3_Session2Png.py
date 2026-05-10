# -*- coding: utf-8 -*-
# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================

import numpy
from PIL import Image
import binascii
import errno    
import os

PNG_SIZE = 28

def getMatrixfrom_pcap(filename,width):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)  
    fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])  
    rn = len(fh)//width
    fh = numpy.reshape(fh[:rn*width],(-1,width))  
    fh = numpy.uint8(fh)
    return fh

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

paths = [['3_ProcessedSession\TrimedSession\Train', '4_Png\Train'],['3_ProcessedSession\TrimedSession\Test', '4_Png\Test']]

for p in paths:
    # Use 'd' (the folder name) instead of 'i' (a number)
    if not os.path.exists(p[0]):
        print(f"Skipping {p[0]} as it does not exist.")
        continue
    for d in os.listdir(p[0]):
        # Keep the actual class name as the folder name
        dir_full = os.path.join(p[1], d)
        mkdir_p(dir_full)
        
        source_dir = os.path.join(p[0], d)
        print(f"Generating PNGs for: {d}")
        
        for f in os.listdir(source_dir):
            bin_full = os.path.join(source_dir, f)
            im = Image.fromarray(getMatrixfrom_pcap(bin_full, PNG_SIZE))
            png_full = os.path.join(dir_full, os.path.splitext(f)[0] + '.png')
            im.save(png_full)