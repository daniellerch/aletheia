
import sys

doc = "\n" \
"  Structural LSB detectors (Statistical attacks to LSB replacement):\n" \
"  - sp:            Sample Pairs Analysis (Octave vesion).\n" \
"  - ws:            Weighted Stego Attack.\n" \
"  - triples:       Triples Attack.\n" \
"  - spa:           Sample Pairs Analysis.\n" \
"  - rs:            RS attack."




# {{{ sp()
def sp():

    if len(sys.argv)!=3:
        print(sys.argv[0], "sp <image>\n")
        sys.exit(0)

    from PIL import Image
    import aletheialib.utils
    import aletheialib.octave_interface as O

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)

    threshold=0.05
    path = aletheialib.utils.absolute_path(sys.argv[2])
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('SP', path, params={"channel":1})["data"][0][0]
        alpha_G = O._attack('SP', path, params={"channel":2})["data"][0][0]
        alpha_B = O._attack('SP', path, params={"channel":3})["data"][0][0]


        if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
            print("No hidden data found")

        if alpha_R>=threshold:
            print("Hidden data found in channel R", alpha_R)
        if alpha_G>=threshold:
            print("Hidden data found in channel G", alpha_G)
        if alpha_B>=threshold:
            print("Hidden data found in channel B", alpha_B)

    else:
        alpha = O._attack('SP', path, params={"channel":1})["data"][0][0]
        if alpha>=threshold:
            print("Hidden data found", alpha)
        else:
            print("No hidden data found")

    sys.exit(0)


    if len(sys.argv)!=3:
        print(sys.argv[0], "spa <image>\n")
        sys.exit(0)

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)

    threshold=0.05
    path = aletheialib.utils.absolute_path(sys.argv[2])
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('SP', path, params={"channel":1})["data"][0][0]
        alpha_G = O._attack('SP', path, params={"channel":2})["data"][0][0]
        alpha_B = O._attack('SP', path, params={"channel":3})["data"][0][0]


        if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
            print("No hidden data found")

        if alpha_R>=threshold:
            print("Hidden data found in channel R", alpha_R)
        if alpha_G>=threshold:
            print("Hidden data found in channel G", alpha_G)
        if alpha_B>=threshold:
            print("Hidden data found in channel B", alpha_B)

    else:
        alpha = O._attack('SP', path, params={"channel":1})["data"][0][0]
        if alpha>=threshold:
            print("Hidden data found", alpha)
        else:
            print("No hidden data found")

    sys.exit(0)
# }}}

# {{{ ws()
def ws(): 
    if len(sys.argv)!=3:
        print(sys.argv[0], "ws <image>\n")
        sys.exit(0)

    from PIL import Image
    import aletheialib.utils
    import aletheialib.octave_interface as O

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)

    threshold=0.05
    path = aletheialib.utils.absolute_path(sys.argv[2])
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('WS', path, params={"channel":1})["data"][0][0]
        alpha_G = O._attack('WS', path, params={"channel":2})["data"][0][0]
        alpha_B = O._attack('WS', path, params={"channel":3})["data"][0][0]

        if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
            print("No hidden data found")

        if alpha_R>=threshold:
            print("Hidden data found in channel R", alpha_R)
        if alpha_G>=threshold:
            print("Hidden data found in channel G", alpha_G)
        if alpha_B>=threshold:
            print("Hidden data found in channel B", alpha_B)

    else:
        alpha = O._attack('WS', path, params={"channel":1})["data"][0][0]
        if alpha>=threshold:
            print("Hidden data found", alpha)
        else:
            print("No hidden data found")

    sys.exit(0)
# }}}

# {{{ triples()
def triples():

    if len(sys.argv)!=3:
        print(sys.argv[0], "triples <image>\n")
        sys.exit(0)

    from PIL import Image
    import aletheialib.utils
    import aletheialib.octave_interface as O

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)


    threshold=0.05
    path = aletheialib.utils.absolute_path(sys.argv[2])
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('TRIPLES', path, params={"channel":1})["data"][0][0]
        alpha_G = O._attack('TRIPLES', path, params={"channel":2})["data"][0][0]
        alpha_B = O._attack('TRIPLES', path, params={"channel":3})["data"][0][0]


        if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
            print("No hidden data found")

        if alpha_R>=threshold:
            print("Hidden data found in channel R", alpha_R)
        if alpha_G>=threshold:
            print("Hidden data found in channel G", alpha_G)
        if alpha_B>=threshold:
            print("Hidden data found in channel B", alpha_B)

    else:
        alpha = O._attack('TRIPLES', path, params={"channel":1})["data"][0][0]
        if alpha>=threshold:
            print("Hidden data found", alpha)
        else:
            print("No hidden data found")

    sys.exit(0)
# }}}

# {{{ aump()
def aump():
    if len(sys.argv)!=3:
        print(sys.argv[0], "aump <image>\n")
        sys.exit(0)

    from PIL import Image
    import aletheialib.utils
    import aletheialib.octave_interface as O

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)


    threshold=0.05
    path = aletheialib.utils.absolute_path(sys.argv[2])
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('AUMP', path, params={"channel":1})["data"][0][0]
        alpha_G = O._attack('AUMP', path, params={"channel":2})["data"][0][0]
        alpha_B = O._attack('AUMP', path, params={"channel":3})["data"][0][0]

        if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
            print("No hidden data found")

        if alpha_R>=threshold:
            print("Hidden data found in channel R", alpha_R)
        if alpha_G>=threshold:
            print("Hidden data found in channel G", alpha_G)
        if alpha_B>=threshold:
            print("Hidden data found in channel B", alpha_B)
    else:
        alpha = O._attack('WS', path, params={"channel":1})["data"][0][0]
        if alpha>=threshold:
            print("Hidden data found", alpha)
        else:
            print("No hidden data found")


    sys.exit(0)
# }}}

# {{{ spa()
def spa():
    if len(sys.argv)!=3:
        print(sys.argv[0], "spa <image>\n")
        sys.exit(0)

    import aletheialib.utils
    import aletheialib.attacks
    from imageio import imread

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)

    threshold=0.05

    I = imread(sys.argv[2])
    if len(I.shape)==2:
        bitrate = aletheialib.attacks.spa_image(I, None)
        if bitrate<threshold:
            print("No hidden data found")
        else:
            print("Hidden data found"), bitrate
    else:
        bitrate_R = aletheialib.attacks.spa_image(I, 0)
        bitrate_G = aletheialib.attacks.spa_image(I, 1)
        bitrate_B = aletheialib.attacks.spa_image(I, 2)

        if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
            print("No hidden data found")
            sys.exit(0)

        if bitrate_R>=threshold:
            print("Hidden data found in channel R", bitrate_R)
        if bitrate_G>=threshold:
            print("Hidden data found in channel G", bitrate_G)
        if bitrate_B>=threshold:
            print("Hidden data found in channel B", bitrate_B)
    sys.exit(0)
# }}}

# {{{ rs()
def rs():
    if len(sys.argv)!=3:
        print(sys.argv[0], "rs <image>\n")
        sys.exit(0)

    import numpy as np
    import aletheialib.utils
    import aletheialib.attacks
    from imageio import imread

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)

    threshold=0.05

    I = np.asarray(imread(sys.argv[2]))
    if len(I.shape)==2:
        bitrate = aletheialib.attacks.rs_image(I)
        if bitrate<threshold:
            print("No hidden data found")
        else:
            print("Hidden data found", bitrate)
    else:
        bitrate_R = aletheialib.attacks.rs_image(I, 0)
        bitrate_G = aletheialib.attacks.rs_image(I, 1)
        bitrate_B = aletheialib.attacks.rs_image(I, 2)

        if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
            print("No hidden data found")
            sys.exit(0)

        if bitrate_R>=threshold:
            print("Hidden data found in channel R", bitrate_R)
        if bitrate_G>=threshold:
            print("Hidden data found in channel G", bitrate_G)
        if bitrate_B>=threshold:
            print("Hidden data found in channel B", bitrate_B)
        sys.exit(0)
# }}}





