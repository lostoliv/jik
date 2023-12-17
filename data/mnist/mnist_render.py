#!/usr/bin/env python3
#
# MNIST Dataset Renderer.
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
  MNIST Dataset Renderer.

  This script generates N random single digit images.
  The digit in each image is randomly sized and placed
  while some distortion and noise gets applied.

  See:
    http://yann.lecun.com/exdb/mnist/

  for the original dataset.
"""


import os, time, random, math, tempfile, argparse, gzip, numpy as np, cv2


def img_formats():
  """
  Available file formats.

  Returns:
    List of available file formats
  """
  return ['png', 'jpg']


def img_default_format():
  """
  Default file format.

  Returns:
    Default file format
  """
  return img_formats()[0]


def create_label_img(res, inv_colors, min_thick_scale, max_thick_scale,
                     min_size_scale, max_size_scale, placement, max_it, seed):
  """
  Create a labeled image.

  Args:
    res            : resolution of the image (square)
    inv_colors     : inverse the colors?
    min_thick_scale: minimum thickness scale
    max_thick_scale: maximum thickness scale
    min_size_scale : minimum size scale
    max_size_scale : maximum size scale
    placement      : placement (0: next to border, 1: centered)
    max_it         : maximum number of iterations
    seed           : random seed (< 0: random value based on system time)

  Returns:
    Image and label (0 to 9)
  """

  # Initialize the seed
  if seed < 0:
    seed = int(time.time())
  random.seed(seed)

  # Available system (OpenCV) fonts
  fonts = [cv2.FONT_HERSHEY_SIMPLEX,
           cv2.FONT_HERSHEY_PLAIN,
           cv2.FONT_HERSHEY_DUPLEX,
           cv2.FONT_HERSHEY_COMPLEX,
           cv2.FONT_HERSHEY_TRIPLEX,
           cv2.FONT_HERSHEY_COMPLEX_SMALL,
           cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
           cv2.FONT_HERSHEY_SCRIPT_COMPLEX]

  # Choose a random font and label
  font  = fonts[random.randint(0, len(fonts) - 1)]
  label = random.randint(0, 9)
  text  = str(label)

  # Choose the right scale and thickness
  min_size = int(min_size_scale * res)
  max_size = int(max_size_scale * res)
  scale    = 1.0
  it       = 0
  while True:
    # Tickness and scale are related
    # Test the result we get with this configuration
    # and if it fits inside the image
    thick = int(random.uniform(min_thick_scale, max_thick_scale) * scale)
    size  = cv2.getTextSize(text, font, scale, thick)[0]
    msize = max(size[0], size[1])

    # Too small
    if msize < min_size:
      scale *= 1.0 + random.random() *\
               (float(min_size - msize) / float(min_size))
      continue

    # Too big
    if msize > max_size:
      scale *= 1.0 - random.random() *\
               (float(msize - max_size) / float(msize))
      continue

    # Good size: we randomize it between the boundaries
    if it > max_it:
      break
    scale *= 1.0 + random.uniform(-float(msize - min_size) / float(msize),
                                   float(max_size - msize) / float(max_size))
    it += 1

  # Randomize the placement
  delta_x = res - size[0]
  delta_x = int((1.0 - placement) * random.randint(-delta_x, delta_x))
  delta_y = res - size[1]
  delta_y = int((1.0 - placement) * random.randint(-delta_y, delta_y))

  # Create the image
  img = np.zeros((res, res), np.int32)

  if inv_colors:
    img[:] = 0xFF
    color  = 0
  else:
    color = 0xFF

  cv2.putText(img, text,
              ((res - size[0] + delta_x) >> 1, (res + size[1] + delta_y) >> 1),
              font, scale, color, thick)
  return (img, label)


def distort_img(img, distortion, seed):
  """
  Distort an image.

  Args:
    img       : image
    distortion: distortion intensity
    seed      : random seed

  Returns:
    Image
  """

  if distortion < np.finfo(np.float32).eps:
    # Nothing to do
    return img

  # Initialize the seed
  if seed < 0:
    seed = int(time.time())
  np.random.seed(seed)

  # Generate a random transformation matrix (identity matrix + some noise)
  mat = np.identity(3) + 0.001 * distortion * (2 * np.random.rand(3, 3) - 1)

  # Apply it
  return cv2.warpPerspective(img, mat, (img.shape[0], img.shape[1]),
                             flags = cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)


def add_gnoise_img(img, gnoise, seed):
  """
  Add gaussian noise to an image.

  Args:
    img   : image
    gnoise: gaussian noise intensity
    seed  : random seed

  Returns:
    Image
  """

  if gnoise < np.finfo(np.float32).eps:
    # Nothing to do
    return img

  # Initialize the seed
  if seed < 0:
    seed = int(time.time())
  np.random.seed(seed)

  # Noise image
  noise = np.random.rand(*img.shape) * gnoise * 0xFF

  # Apply gaussian noise
  return img + noise


def add_spnoise_img(img, spnoise, seed):
  """
  Add salt-pepper noise to an image.

  Args:
    img    : image
    spnoise: salt-pepper noise intensity
    seed   : random seed

  Returns:
    Image
  """

  if spnoise < np.finfo(np.float32).eps:
    # Nothing to do
    return img

  # Initialize the seed
  if seed < 0:
    seed = int(time.time())
  np.random.seed(seed)

  # Noise image and threshold
  noise     = np.random.rand(*img.shape) * 0xFF
  threshold = spnoise * 0.1 * 0xFF

  # Apply salt-pepper noise
  res = img.copy()
  res[noise < threshold]        = 0
  res[noise > 0xFF - threshold] = 0xFF

  return res


def render_img(res, inv_colors, min_thick_scale, max_thick_scale,
               min_size_scale, max_size_scale, min_distortion, max_distortion,
               min_gaussian_noise, max_gaussian_noise, min_salt_pepper_noise,
               max_salt_pepper_noise, placement, max_it, seed):
  """
  Render an image.

  Args:
    res                  : resolution of the image (square)
    inv_colors           : inverse the colors?
    min_thick_scale      : minimum thickness scale
    max_thick_scale      : maximum thickness scale
    min_size_scale       : minimum size scale
    max_size_scale       : maximum size scale
    min_distortion       : minimum distortion
    max_distortion       : maximum distortion
    min_gaussian_noise   : minimum gaussian noise
    max_gaussian_noise   : maximum gaussian noise
    min_salt_pepper_noise: minimum salt-pepper noise
    max_salt_pepper_noise: maximum salt-pepper noise
    placement            : placement (0: next to border, 1: centered)
    max_it               : maximum number of iterations
    seed                 : random seed (< 0: random value based on system time)

  Returns:
    Image and label (0 to 9)
  """

  # Initialize the seed
  if seed < 0:
    seed = int(time.time())
  random.seed(seed)

  # Random values for distortion, gaussian and salt-pepper noises
  distortion        = random.uniform(min_distortion, max_distortion)
  gaussian_noise    = random.uniform(min_gaussian_noise, max_gaussian_noise)
  salt_pepper_noise = random.uniform(min_salt_pepper_noise,
                                     max_salt_pepper_noise)

  # Create the label image
  (img, label) = create_label_img(res, inv_colors, min_thick_scale,
                                  max_thick_scale, min_size_scale,
                                  max_size_scale, placement, max_it, seed + 1)

  # Distort the image
  img = distort_img(img, distortion, seed + 2)

  # Add gaussian noise to the image
  img = add_gnoise_img(img, gaussian_noise, seed + 3)

  # Add salt-pepper noise to the image
  img = add_spnoise_img(img, salt_pepper_noise, seed + 4)

  # Convert the image to 8-bits if needed
  if img.dtype == np.uint8:
    img8 = img
  else:
    img8 = np.zeros(img.shape, np.uint8)
    cv2.convertScaleAbs(img, img8)

  return (img8, label)


def int32(val):
  """
  Convert an integer value to a 32-bit bitarray.

  Args:
    val: integer value

  Returns:
    32-bit bitarray
  """
  return bytearray([(val >> i & 0xFF) for i in (24, 16, 8, 0)])


def render(num_img, res, inv_colors, min_thick_scale, max_thick_scale,
           min_size_scale, max_size_scale, min_distortion, max_distortion,
           min_gaussian_noise, max_gaussian_noise, min_salt_pepper_noise,
           max_salt_pepper_noise, placement, max_it, seed, format, output_dir,
           prefix, concatenate, dataset, gz):
  """
  Render some images.

  Args:
    num_img              : number of images to render
    res                  : resolution of the image (square)
    inv_colors           : inverse the colors?
    min_thick_scale      : minimum thickness scale
    max_thick_scale      : maximum thickness scale
    min_size_scale       : minimum size scale
    max_size_scale       : maximum size scale
    min_distortion       : minimum distortion
    max_distortion       : maximum distortion
    min_gaussian_noise   : minimum gaussian noise
    max_gaussian_noise   : maximum gaussian noise
    min_salt_pepper_noise: minimum salt-pepper noise
    max_salt_pepper_noise: maximum salt-pepper noise
    placement            : placement (0: next to border, 1: centered)
    max_it               : maximum number of iterations
    seed                 : random seed (< 0: random value based on system time)
    format               : output file format
    output_dir           : output directory
    prefix               : file prefix
    concatenate          : concatenate all the images
    dataset              : create a dataset
    gz                   : gzip the dataset?
  """

  if format not in img_formats():
    print(f'Unknown file format {format}')
    return

  # Check the output directory exists, create it if not
  if not os.path.isdir(output_dir):
    try:
      os.makedirs(output_dir)
    except Exception as e:
      print(f'Cannot create {output_dir}: {e}')
      return

  # We need to generate at least 1 image
  if num_img < 1:
    num_img = 1

  if not prefix:
    prefix = 'mnist'

  if dataset:
    concatenate = False

  if concatenate:
    bigmat_cols = int(math.floor(math.sqrt(num_img) + 0.5))
    col_imgs    = []
    row_imgs    = []
    shape       = None

  if dataset:
    # Create the MNIST files
    image_path = os.path.join(output_dir, f'{prefix}-images-idx3-ubyte')
    label_path = os.path.join(output_dir, f'{prefix}-labels-idx1-ubyte')
    try:
      if gz:
        image_path += '.gz'
        label_path += '.gz'
        image_file = tempfile.NamedTemporaryFile(delete = False)
        label_file = tempfile.NamedTemporaryFile(delete = False)
      else:
        image_file = open(image_path, 'wb')
        label_file = open(label_path, 'wb')
    except Exception as e:
      print(f'Cannot write output MNIST files: {e}')
      return
    # Magic number
    image_file.write(int32(0x803))
    label_file.write(int32(0x801))
    # Number of images
    image_file.write(int32(num_img))
    label_file.write(int32(num_img))
    # Image width and height
    image_file.write(int32(res))
    image_file.write(int32(res))

  for i in range(num_img):
    # Render the image
    (img, label) = render_img(res, inv_colors, min_thick_scale,
                              max_thick_scale, min_size_scale, max_size_scale,
                              min_distortion, max_distortion,
                              min_gaussian_noise, max_gaussian_noise,
                              min_salt_pepper_noise, max_salt_pepper_noise,
                              placement, max_it, seed + i)
    if dataset:
      image_file.write(img.tobytes())
      label_file.write(bytearray([label]))
      image_file.flush()
      label_file.flush()
      continue

    if concatenate:
      # Append images to the list of column images
      ishape = img.shape
      if not shape:
        shape = ishape
      if ishape != shape:
        print(f'Images have different shapes ({ishape} instead of {shape})')
        return
      col_imgs.append(img)
      if len(col_imgs) >= bigmat_cols:
        # Once the colum is complete, stack all the images and
        # append to result images to the list of row images
        row_imgs.append(np.hstack(col_imgs))
        col_imgs = []
      continue

    # Just save the image
    img_name = f'{prefix}_{i:06d}.{format}'
    img_path = os.path.join(output_dir, img_name)
    if not cv2.imwrite(img_path, img):
      print(f'Cannot create image file {img_path}')
      return

  if concatenate:
    if len(col_imgs):
      # Append black images to complete the column
      for i in range(len(col_imgs), bigmat_cols):
        col_imgs.append(np.zeros(shape, np.uint8))
      row_imgs.append(np.hstack(col_imgs))
    # Stack all the row images
    img = np.vstack(row_imgs)
    # Save the final image
    img_path = os.path.join(output_dir, f'{prefix}.{format}')
    if not cv2.imwrite(img_path, img):
      print(f'Cannot create image file {img_path}')
      return

  if dataset:
    # Cleanup
    if gz:
      image_path_tmp = image_file.name
      label_path_tmp = label_file.name
    image_file.close()
    label_file.close()
    if gz:
      # Gzip the file
      try:
        image_file    = open(image_path_tmp, 'rb')
        label_file    = open(label_path_tmp, 'rb')
        image_file_gz = gzip.open(image_path, 'wb')
        label_file_gz = gzip.open(label_path, 'wb')
      except Exception as e:
        print('Cannot write output MNIST files: {e}')
        return
      image_file_gz.write(image_file.read())
      label_file_gz.write(label_file.read())
      # Cleanup
      image_file.close()
      label_file.close()
      image_file_gz.close()
      label_file_gz.close()
      os.remove(image_path_tmp)
      os.remove(label_path_tmp)


def main():
  """
  Main function.
  """

  # Arguments parser
  parser = argparse.ArgumentParser(description = 'MNIST Dataset Renderer.')
  parser.add_argument('-out'    , help = 'Output directory',
                                  type = str, required = True)
  parser.add_argument('-num'    , help = 'Number of images to render',
                                  type = int, default = 1)
  parser.add_argument('-res'    , help = 'Resolution of the rendered images',
                                  type = int, default = 28)
  parser.add_argument('-inv'    , help = 'Inverse the colors',
                                  action = 'store_true')
  parser.add_argument('-tmin'   , help = 'Min thickness scale',
                                  type = float, default = 0.8)
  parser.add_argument('-tmax'   , help = 'Max thickness scale',
                                  type = float, default = 1.2)
  parser.add_argument('-fmin'   , help = 'Min font scale',
                                  type = float, default = 0.6)
  parser.add_argument('-fmax'   , help = 'Max font scale',
                                  type = float, default = 0.8)
  parser.add_argument('-gnmin'  , help = 'Min gaussian noise',
                                  type = float, default = 0.0)
  parser.add_argument('-gnmax'  , help = 'Max gaussian noise',
                                  type = float, default = 0.0)
  parser.add_argument('-spnmin' , help = 'Min salt-pepper noise',
                                  type = float, default = 0.0)
  parser.add_argument('-spnmax' , help = 'Max salt-pepper noise',
                                  type = float, default = 0.0)
  parser.add_argument('-dmin'   , help = 'Min distortion',
                                  type = float, default = 0.0)
  parser.add_argument('-dmax'   , help = 'Max distortion',
                                  type = float, default = 0.0)
  parser.add_argument('-place'  , help = 'Placement',
                                  type = float, default = 0.0)
  parser.add_argument('-it'     , help = 'Max number of iterations',
                                  type = int, default = 10)
  parser.add_argument('-seed'   , help = 'Random seed',
                                  type = int, default = -1)
  parser.add_argument('-format' , help = 'Image file format (%s)'.format(
                                  ', '.join(img_formats())),
                                  type = str, default = img_default_format())
  parser.add_argument('-prefix' , help = 'Filename prefix', default = 'mnist')
  parser.add_argument('-concat' , help = 'Concatenate all the images into one',
                                  action = 'store_true')
  parser.add_argument('-dataset', help = 'Create a MNIST dataset',
                                  action = 'store_true')
  parser.add_argument('-gz'     , help = 'Gzip the MNIST dataset',
                                  action = 'store_true')

  args = parser.parse_args()

  # Call the renderer
  render(args.num, args.res, args.inv, args.tmin, args.tmax, args.fmin,
         args.fmax, args.dmin, args.dmax, args.gnmin, args.gnmax,
         args.spnmin, args.spnmax, args.place, args.it, args.seed,
         args.format, args.out, args.prefix, args.concat, args.dataset,
         args.gz)


if __name__ == '__main__':
  main()
