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
"  - plot-dct-histogram:    Plot DCT histogram.\n" \
"  - eof-extract:           Extract the data after EOF.\n" \
"  - print-metadata:        Print Exif metadata.\n" \
"  - download-external:     Download all the external code (Octave, F5, e4s).\n" \


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
        sys.exit(1)
    if not os.path.isfile(stego):
        print("Stego file not found:", stego)
        sys.exit(1)

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
        sys.exit(1)
    if not os.path.isfile(stego):
        print("Stego file not found:", stego)
        sys.exit(1)


    name, ext = os.path.splitext(cover)
    if ext.lower() not in [".jpeg", ".jpg"] or not os.path.isfile(cover):
        print("Please, provide a JPEG image!\n")
        sys.exit(1)

    name, ext = os.path.splitext(stego)
    if ext.lower() not in [".jpeg", ".jpg"] or not os.path.isfile(stego):
        print("Please, provide a JPEG image!\n")
        sys.exit(1)



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
    if len(I.shape) == 2:
        data.append(I.flatten())
    elif len(I.shape) == 3:
        for i in range(I.shape[2]):
            data.append(I[:,:,i].flatten())

    plt.hist(data, bins=256, range=(0, 256), 
            color=["r", "g", "b"][:len(data)], 
            label=["R", "G", "B"][:len(data)])
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
        sys.exit(1)
    I = aletheialib.jpeg.JPEG(fn)
    channels = ["r", "g", "b"]
    dct_list = []
    for i in range(I.components()):
        dct = I.coeffs(i).flatten()
        dct_list.append(dct)
        #counts, bins = np.histogram(dct, range(-5, 5))
        #plt.plot(bins[:-1], counts, channels[i])
    plt.hist(dct_list, range(-100, 100), rwidth=1, color=["r", "g", "b"][:len(dct_list)])

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
        sys.exit(1)

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
        sys.exit(1)

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
        sys.exit(1)

    import aletheialib.attacks
    aletheialib.attacks.exif(sys.argv[2])
    sys.exit(0)
# }}}



# {{{ download_external
def download_external():

    if len(sys.argv)>3 or (len(sys.argv)==3 and sys.argv[2]!="--status"):
        print(sys.argv[0], "download-external [--status]\n")
        print("  Download all the external code used by Aletheia. The license of")
        print("  each resource is shown and must be accepted before downloading it.")
        print("  Resources already in the cache are skipped.\n")
        print("  --status:   Only show which resources are already downloaded.")
        print("")
        sys.exit(0)

    import aletheialib.utils as u

    currdir = os.path.dirname(u.__file__)
    cache_dir = os.path.join(os.path.dirname(currdir), 'aletheia-cache')
    octave_dir = os.path.join(cache_dir, 'octave')

    octave_code = [
        "WS", "TRIPLES", "AUMP",
        "SRM", "SRMQ1", "SCRMQ1", "GFR", "DCTR",
        "HUGO", "WOW", "HILL", "HILL_COLOR", "S_UNIWARD", "S_UNIWARD_COLOR",
        "J_UNIWARD", "J_UNIWARD_COLOR", "J_MIPOD", "J_MIPOD_COLOR",
        "NSF5", "NSF5_COLOR", "EBS", "EBS_COLOR", "UED", "UED_COLOR",
    ]

    # (name, file that marks it as downloaded, download function)
    # The JPEG toolbox goes first, since the JPEG methods need it.
    resources = [
        ("JPEG_TOOLBOX", os.path.join(cache_dir, 'jpeg_toolbox', 'Makefile'),
            u.download_octave_jpeg_toolbox),
    ]
    for name in octave_code:
        resources.append((name, os.path.join(octave_dir, name+'.m'),
            lambda name=name: u.download_octave_code(name)))

    # Data file needed by J-MIPOD, only if its code has been accepted
    def download_ixlnx():
        if (os.path.isfile(os.path.join(octave_dir, 'J_MIPOD.m')) or
            os.path.isfile(os.path.join(octave_dir, 'J_MIPOD_COLOR.m'))):
            u.download_octave_aux_file("ixlnx3_5.mat")
        else:
            print("Skipped: J_MIPOD has not been downloaded")

    resources += [
        ("ixlnx3_5.mat", os.path.join(octave_dir, 'ixlnx3_5.mat'),
            download_ixlnx),
        ("E4S", os.path.join(octave_dir, 'ensemble_training.m'),
            u.download_e4s),
        ("F5", os.path.join(cache_dir, 'F5', 'Extract.class'),
            u.download_F5),
    ]

    if len(sys.argv)<3:
        for name, path, download in resources:
            if os.path.isfile(path):
                continue
            print("\n==== "+name+" ====")
            # The download functions call sys.exit() when the license is
            # not accepted or the download fails: skip and go on.
            try:
                download()
            except SystemExit:
                pass

    print("\nExternal resources:")
    missing = 0
    for name, path, _ in resources:
        if os.path.isfile(path):
            print("  [x]", name)
        else:
            print("  [ ]", name)
            missing += 1
    if missing:
        print("\n", missing, "resources not downloaded\n")
    else:
        print("\nAll the external resources are downloaded\n")
    sys.exit(0)
# }}}
