import sys

doc = "\n" \
"  Tools:\n" \
"  - hpf:                   High-pass filter.\n" \
"  - print-diffs:           Differences between two images.\n" \
"  - print-dct-diffs:       Differences between the DCT coefficients of two JPEG images.\n" \
"  - rm-alpha:              Opacity of the alpha channel to 255.\n" \
"  - plot-histogram:        Plot histogram.\n" \
"  - plot-histogram-diff:   Plot histogram of differences.\n" \
"  - plot-dct-histogram:    Plot DCT histogram.\n" \


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

    if len(sys.argv)<2:
        print(sys.argv[0], "plot-histogram <image>\n")
        print("")
        sys.exit(0)

    import aletheialib.utils
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

# {{{ plot_histogram_diff
def plot_histogram_diff():

    if len(sys.argv)<4:
        print(sys.argv[0], "plot-histogram <image> <L|R|U|D>\n")
        print("")
        sys.exit(0)

    import aletheialib.utils
    fn = aletheialib.utils.absolute_path(sys.argv[2])
    direction = sys.argv[3]
    if direction not in ["L", "R", "U", "D"]:
        print("Please provide the substract direction: L, R, U or D\n")
        sys.exit(0)

    I = imageio.imread(fn)
    data = []
    if len(I.shape) == 1:
        if direction == "L":
            D = I[:,1:]-I[:,:-1]
        if direction == "R":
            D = I[:,:-1]-I[:,1:]
        if direction == "U":
            D = I[:-1,:]-I[1:,:]
        if direction == "D":
            D = I[1:,:]-I[:-1,:]
        
        data.append(D.flatten())
    else:
        for i in range(I.shape[2]):
            if direction == "L":
                D = I[:,1:,i]-I[:,:-1,i]
            if direction == "R":
                D = I[:,:-1,i]-I[:,1:,i]
            if direction == "U":
                D = I[:-1,:,i]-I[1:,:,i]
            if direction == "D":
                D = I[1:,:,i]-I[:-1,:,i]

            data.append(D.flatten())

    plt.hist(data, range(0, 255), color=["r", "g", "b"])
    plt.show()
    sys.exit(0)

# }}}

# {{{ plot_dct_histogram
def plot_dct_histogram():

    if len(sys.argv)<2:
        print(sys.argv[0], "plot-dct-histogram <image>\n")
        print("")
        sys.exit(0)

    import aletheialib.utils
    import aletheialib.jpeg

    fn = aletheialib.utils.absolute_path(sys.argv[2])
    name, ext = os.path.splitext(fn)
    if ext.lower() not in [".jpeg", ".jpg"] or not os.path.isfile(fn):
        print("Please, provide a a JPEG image!\n")
        sys.exit(0)
    I = aletheialib.jpeg.JPEG(fn)
    channels = ["r", "g", "b"]
    dct_list = []
    for i in range(I.components()):
        dct = I.coeffs(i).flatten()
        dct_list.append(dct)
        #counts, bins = np.histogram(dct, range(-5, 5))
        #plt.plot(bins[:-1], counts, channels[i])
    plt.hist(dct_list, range(-5, 5), rwidth=1, color=["r", "g", "b"])

    plt.show()
    sys.exit(0)
# }}}


