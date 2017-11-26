# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import argparse
import os
import io
import shutil
import codecs
import sys

# zero based
# smile flag at 31th position
g_smile_index = 31
g_neutral_face_path = "neutral_face"

random.seed(1)
parser = argparse.ArgumentParser()
parser.add_argument("--celeba_path", help="specify celeba path", required=True)
parser.add_argument("--celeba_attr", help="specify celeba attribute file", required=True)
parser.add_argument("--output_path", help="specify output path", required=True)
args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
if not os.path.exists(g_neutral_face_path):
    os.makedirs(g_neutral_face_path)

with codecs.open(os.path.join(args.output_path, "filelist.txt"), 'w', 'utf-8') as f, \
    codecs.open(os.path.join(g_neutral_face_path, "filelist.txt"), 'w', 'utf-8') as g:
    for filelineno, line in enumerate(io.open(args.celeba_attr, encoding="utf-8")):
        if filelineno < 2:
            continue
        line_split = line.strip().split()
        #print(line)
        # line_split[0] is filename, attribute index start at one
        filename = line_split[0]
        smile = line_split[g_smile_index + 1]
        #print(smile)
        if int(smile) == 1:
            shutil.copy(os.path.join(args.celeba_path, filename), args.output_path)
            f.write(line)
        else:
            shutil.copy(os.path.join(args.celeba_path, filename), g_neutral_face_path)
            g.write(line)
        if filelineno % 1000 == 0:
            sys.stdout.write("*")
            sys.stdout.flush()
print("")
