import os
import sys

doc = "\n" \
"  Tools:\n" \
"  - hpf:                   High-pass filter.\n" \
"  - print-diffs:           Differences between two images.\n" \
"  - print-dct-diffs:       Differences between the DCT coefficients of two JPEG images.\n" \
"  - print-pixels:          Print a range of píxels.\n" \
"  - print-coeffs:          Print a range of JPEG coefficients.\n" \
"  - rm-alpha:              Opacity of the alpha channel to 255.\n" \
"  - plot-histogram:        Plot histogram.\n" \
"  - plot-histogram-diff:   Plot histogram of differences.\n" \
"  - plot-dct-histogram:    Plot DCT histogram.\n" \
"  - eof-extrat:            Extract the data after EOF.\n" \
"  - print-metadata:        Print Exif metadata.\n" \


# {{{ hpf
def hpf():

    if len(sys.argv)!=4:
        print(sys.argv[0], "hpf <input-image> <output-image>\n")
        print("")
        sys.exit(0)

    import aletheialib.attacks
    aletheialib.attacks.high_pass_filter(sys.argv[2], sys.argv[3])
    sys.exit(0)
# }}}

# {{{ print_diffs
def print_diffs():

    if len(sys.argv)!=4:
        print(sys.argv[0], "print-diffs <cover image> <stego image>\n")
        print("")
        sys.exit(0)

    import aletheialib.utils
    import aletheialib.attacks

    cover = aletheialib.utils.absolute_path(sys.argv[2])
    stego = aletheialib.utils.absolute_path(sys.argv[3])
    if not os.path.isfile(cover):
        print("Cover file not found:", cover)
        sys.exit(0)
    if not os.path.isfile(stego):
        print("Stego file not found:", stego)
        sys.exit(0)

    aletheialib.attacks.print_diffs(cover, stego)
    sys.exit(0)
# }}}

# {{{ print_dct_diffs
def print_dct_diffs():

    if len(sys.argv)!=4:
        print(sys.argv[0], "print-dtc-diffs <cover image> <stego image>\n")
        print("")
        sys.exit(0)

    import aletheialib.attacks
    import aletheialib.utils

    cover = aletheialib.utils.absolute_path(sys.argv[2])
    stego = aletheialib.utils.absolute_path(sys.argv[3])

    if not os.path.isfile(cover):
        print("Cover file not found:", cover)
        sys.exit(0)
    if not os.path.isfile(stego):
        print("Stego file not found:", stego)
        sys.exit(0)


    name, ext = os.path.splitext(cover)
    if ext.lower() not in [".jpeg", ".jpg"] or not os.path.isfile(cover):
        print("Please, provide a JPEG image!\n")
        sys.exit(0)

    name, ext = os.path.splitext(stego)
    if ext.lower() not in [".jpeg", ".jpg"] or not os.path.isfile(stego):
        print("Please, provide a JPEG image!\n")
        sys.exit(0)



    aletheialib.attacks.print_dct_diffs(cover, stego)
    sys.exit(0)
# }}}

# {{{ rm_alpha
def rm_alpha():

    if len(sys.argv)!=4:
        print(sys.argv[0], "rm-alpha <input-image> <output-image>\n")
        print("")
        sys.exit(0)

    import aletheialib.utils
    import aletheialib.attacks

    aletheialib.attacks.remove_alpha_channel(sys.argv[2], sys.argv[3])
    sys.exit(0)
# }}}

# {{{ plot_histogram
def plot_histogram():

    if len(sys.argv)<3:
        print(sys.argv[0], "plot-histogram <image>\n")
        print("")
        sys.exit(0)

    import imageio
    import aletheialib.utils
    from matplotlib import pyplot as plt

    fn = aletheialib.utils.absolute_path(sys.argv[2])
    I = imageio.imread(fn)
    data = []
    if len(I.shape) == 1:
        data.append(I.flatten())
    else:
        for i in range(I.shape[2]):
            data.append(I[:,:,i].flatten())

    plt.hist(data, range(0, 255), color=["r", "g", "b"])
    plt.show()
    sys.exit(0)

# }}}

# {{{ plot_dct_histogram
def plot_dct_histogram():

    if len(sys.argv)<3:
        print(sys.argv[0], "plot-dct-histogram <image>\n")
        print("")
        sys.exit(0)

    import aletheialib.utils
    import aletheialib.jpeg
    from matplotlib import pyplot as plt

    fn = aletheialib.utils.absolute_path(sys.argv[2])
    name, ext = os.path.splitext(fn)
    if ext.lower() not in [".jpeg", ".jpg"] or not os.path.isfile(fn):
        print("Please, provide a JPEG image!\n")
        sys.exit(0)
    I = aletheialib.jpeg.JPEG(fn)
    channels = ["r", "g", "b"]
    dct_list = []
    for i in range(I.components()):
        dct = I.coeffs(i).flatten()
        dct_list.append(dct)
        #counts, bins = np.histogram(dct, range(-5, 5))
        #plt.plot(bins[:-1], counts, channels[i])
    plt.hist(dct_list, range(-10, 10), rwidth=1, color=["r", "g", "b"])

    plt.show()
    sys.exit(0)
# }}}

# {{{ print_pixels()
def print_pixels():

    if len(sys.argv)!=5:
        print(sys.argv[0], "print-pixels <image> <width start>:<width end> <height start>:<height end>\n")
        print("Example:")
        print(sys.argv[0], "print-pixels test.png 400:410 200:220\n")
        print("")
        sys.exit(0)

    import imageio
    I = imageio.imread(sys.argv[2])

    w = sys.argv[3].split(":")
    h = sys.argv[4].split(":")
    ws = int(w[0])
    we = int(w[1])
    hs = int(h[0])
    he = int(h[1])


    if len(I.shape) == 2:
        print("Image shape:", I.shape[:2])
        print(I[hs:he, ws:we])
    else:
        print("Image shape:", I.shape[:2])
        for ch in range(I.shape[2]):
            print("Channel:", ch)
            print(I[hs:he, ws:we, ch])
            print()

# }}}

# {{{ print_coeffs()
def print_coeffs():

    if len(sys.argv)!=5:
        print(sys.argv[0], "print-coeffs <image> <width start>:<width end> <height start>:<height end>\n")
        print("Example:")
        print(sys.argv[0], "print-coeffs test.jpg 400:410 200:220\n")
        print("")
        sys.exit(0)

    w = sys.argv[3].split(":")
    h = sys.argv[4].split(":")
    ws = int(w[0])
    we = int(w[1])
    hs = int(h[0])
    he = int(h[1])


    fn, ext = os.path.splitext(sys.argv[2])
    if ext[1:].lower() not in ["jpg", "jpeg"]:
        print("ERROR: Please, provide a JPEG image")
        sys.exit(0)

    import aletheialib.utils
    from aletheialib.jpeg import JPEG

    img = aletheialib.utils.absolute_path(sys.argv[2])
    im_jpeg = JPEG(img)

    for i in range(im_jpeg.components()):
        coeffs = im_jpeg.coeffs(i)
        print("Image shape:", coeffs.shape)
        print("Channel:", i)
        print(coeffs[hs:he, ws:we])
        print()


# }}}

# {{{ eof_extract
def eof_extract():

    if len(sys.argv)!=4:
        print(sys.argv[0], "eof-extract <input-image> <output-data>\n")
        print("")
        sys.exit(0)

    if not os.path.isfile(sys.argv[2]):
        print("Please, provide a valid image!\n")

    import aletheialib.attacks
    aletheialib.attacks.eof_extract(sys.argv[2], sys.argv[3])
    sys.exit(0)
# }}}

# {{{ print_metadata
def print_metadata():

    if len(sys.argv)!=3:
        print(sys.argv[0], "print-metadata <input-image>\n")
        print("")
        sys.exit(0)

    if not os.path.isfile(sys.argv[2]):
        print("Please, provide a valid image!\n")

    import aletheialib.attacks
    aletheialib.attacks.exif(sys.argv[2])
    sys.exit(0)
# }}}


