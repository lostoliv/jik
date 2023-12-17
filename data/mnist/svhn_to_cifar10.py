#!/usr/bin/env python3
#
# CIFAR-10 Dataset Renderer.
#
# The MIT License (MIT)
#
# Copyright (c)2016 Olivier Soares
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


"""
  SVHN to CIFAR-10 Dataset Converter.

  This script converts a SVHN dataset (Street View House Numbers)
  to a CIFAR-10 dataset format.

  See the SVHN dataset format here:
    http://ufldl.stanford.edu/housenumbers

  See the CIFAR-10 dataset format here:
    https://www.cs.toronto.edu/~kriz/cifar.html
"""


import os, array, argparse, numpy as np, scipy.io as spio, cv2


def svhn_to_cifar10(svhn_path, output_dir, res, prefix):
  """
  Convert a SVHN dataset to a CIFAR-10 dataset format.

  Args:
    svhn_path : path to the svhn mat file
    output_dir: output directory to store the CIFAR-10 dataset
    res       : output CIFAR-10 dataset resolution
    prefix    : prefix to the output CIFAR-10 dataset
  """

  if not prefix:
    prefix = 'cifar10'

  try:
    mat    = spio.loadmat(svhn_path)
    pixels = mat['X']
    labels = mat['y']
  except Exception as e:
    print(f'Cannot read {svhn_path}: {e}')
    return

  # Input parameters
  num_img        = len(labels)
  img_src_width  = len(pixels)
  img_src_height = len(pixels[0])
  img_src_depth  = len(pixels[0][0])
  img_src_size   = img_src_width * img_src_height

  # Output parameters
  img_dst_width  = res
  img_dst_height = res

  if len(pixels[0][0][0]) != num_img:
    print(f'Invalid input file {input_file}')
    return

  # Check the output directory exists, create it if not
  if not os.path.isdir(output_dir):
    try:
      os.makedirs(output_dir)
    except Exception as e:
      print(f'Cannot create {output_dir}: {e}')
      return

  # Path to the image and label files
  image_path = os.path.join(output_dir, f'{prefix}_batch.bin')

  # Create the CIFAR-10 files
  try:
    image_file = open(image_path, 'wb')
  except Exception as e:
    print(f'Cannot write output CIFAR-10 file: {e}')
    return

  print(f'Encoding {num_img} image(s) from {svhn_path} to {image_path}')

  perc      = 0
  perc_iter = 10
  perc_next = 10
  images    = []
  for i in range(num_img):
    perc = int(100 * float(i + 1) / num_img)
    if perc >= perc_next:
      print(f'{perc}% done')
      perc_next += perc_iter
    # Read the image from the mat, resize it and add it to the list
    img = np.zeros((img_src_width, img_src_height, 3), 'uint8')
    if img_src_depth == 1:
      for y in range(img_src_height):
        for x in range(img_src_width):
          img[x][y][0] = pixels[x][y][0][i]
          img[x][y][1] = img[x][y][0]
          img[x][y][2] = img[x][y][0]
    elif img_src_depth == 3:
      for y in range(img_src_height):
        for x in range(img_src_width):
          img[x][y][0] = pixels[x][y][0][i]
          img[x][y][1] = pixels[x][y][1][i]
          img[x][y][2] = pixels[x][y][2][i]
    else:
      print(f'Unknown image depth of {img_src_depth}')
      return
    img = cv2.resize(img, (img_dst_width, img_dst_height))
    # Label is saved as 1-based, convert to 0-based
    image_file.write(bytearray([labels[i][0] - 1]))
    img.tofile(image_file)
    image_file.flush()

  # Cleanup
  image_file.close()


def main():
  """
  Main function.
  """

  # Arguments parser
  parser = argparse.ArgumentParser(description = 'SVHN to CIFAR-10 Dataset.')
  parser.add_argument('-svhn', help = 'Input SVHN',
                      type = str, required = True)
  parser.add_argument('-out', help = 'Output directory',
                      type = str, required = True)
  parser.add_argument('-res', help = 'Resolution of the output images',
                      type = int, default = 32)
  parser.add_argument('-prefix', help = 'Output prefix', type = str)

  args = parser.parse_args()

  # Converter
  svhn_to_cifar10(args.svhn, args.out, args.res, args.prefix)


if __name__ == '__main__':
  main()
