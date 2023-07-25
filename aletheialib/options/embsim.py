import sys
import os
from aletheialib.utils import download_octave_code, download_octave_jpeg_toolbox

doc="\n" \
"  Embedding simulators:\n" \
"  - lsbr-sim:             LSB replacement simulator.\n" \
"  - lsbm-sim:             LSB matching simulator.\n" \
"  - hugo-sim:             HUGO simulator.\n" \
"  - wow-sim:              WOW simulator.\n" \
"  - s-uniward-sim:        Spatial UNIWARD simulator.\n" \
"  - s-uniward-color-sim:  Spatial UNIWARD color simulator.\n" \
"  - j-uniward-sim:        JPEG UNIWARD simulator.\n" \
"  - j-uniward-color-sim:  JPEG UNIWARD color simulator.\n" \
"  - hill-sim:             HILL simulator.\n" \
"  - hill-color-sim:       HILL color simulator.\n" \
"  - ebs-sim:              EBS simulator.\n" \
"  - ebs-color-sim:        EBS color simulator.\n" \
"  - ued-sim:              UED simulator.\n" \
"  - ued-color-sim:        UED color simulator.\n" \
"  - nsf5-sim:             nsF5 simulator.\n" \
"  - nsf5-color-sim:       nsF5 color simulator.\n" \
"  - steghide-sim:         Steghide simulator.\n" \
"  - outguess-sim:         Outguess simulator.\n" \
"  - steganogan-sim:       SteganoGAN simulator." 




# {{{ lsbr
def lsbr():

    if len(sys.argv)!=5:
        print(sys.argv[0], "lsbr-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.lsbr, 
                                       sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0);
# }}}

# {{{ lsbm
def lsbm():

    if len(sys.argv)!=5:
        print(sys.argv[0], "lsbm-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.lsbm, 
                                       sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0);
# }}}

# {{{ hugo
def hugo():

    if len(sys.argv)!=5:
        print(sys.argv[0], "hugo-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_code("HUGO")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.hugo, 
                                       sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0);
# }}}

# {{{ wow
def wow():

    if len(sys.argv)!=5:
        print(sys.argv[0], "wow-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("WOW")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.wow, 
                                       sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0);
# }}}

# {{{ s_uniward
def s_uniward():

    if len(sys.argv)!=5:
        print(sys.argv[0], "s-uniward-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_code("S_UNIWARD")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.s_uniward, 
                                       sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0);
# }}}

# {{{ s_uniward_color
def s_uniward_color():

    if len(sys.argv)!=5:
        print(sys.argv[0], "s-uniward-color-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_code("S_UNIWARD_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.s_uniward_color, 
                                       sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0);
# }}}

# {{{ hill
def hill():

    if len(sys.argv)!=5:
        print(sys.argv[0], "hill-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_code("HILL")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.hill, 
                                       sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0);
# }}}

# {{{ hill_color
def hill_color():

    if len(sys.argv)!=5:
        print(sys.argv[0], "hill-color-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_code("HILL_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.hill_color, 
                                       sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0);
# }}}

# {{{ j_uniward
def j_uniward():

    if len(sys.argv)!=5:
        print(sys.argv[0], "j-uniward-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("J_UNIWARD")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.j_uniward, 
            sys.argv[2], sys.argv[3], sys.argv[4], embed_fn_saving=True)
    sys.exit(0);
# }}}

# {{{ j_uniward_color
def j_uniward_color():

    if len(sys.argv)!=5:
        print(sys.argv[0], "j-uniward-color-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("J_UNIWARD_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.j_uniward_color, 
            sys.argv[2], sys.argv[3], sys.argv[4], embed_fn_saving=True)
    sys.exit(0);
# }}}

# {{{ ebs
def ebs():

    if len(sys.argv)!=5:
        print(sys.argv[0], "ebs-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("EBS")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.ebs, 
            sys.argv[2], sys.argv[3], sys.argv[4], embed_fn_saving=True)
    sys.exit(0);
# }}}

# {{{ ebs-color
def ebs_color():

    if len(sys.argv)!=5:
        print(sys.argv[0], "ebs-color-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("EBS_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.ebs_color, 
            sys.argv[2], sys.argv[3], sys.argv[4], embed_fn_saving=True)
    sys.exit(0);
# }}}

# {{{ ued
def ued():

    if len(sys.argv)!=5:
        print(sys.argv[0], "ued-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("UED")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.ued, 
            sys.argv[2], sys.argv[3], sys.argv[4], embed_fn_saving=True)
    sys.exit(0);
# }}}

# {{{ ued-color
def ued_color():

    if len(sys.argv)!=5:
        print(sys.argv[0], "ued-color-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("UED_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.ued_color, 
            sys.argv[2], sys.argv[3], sys.argv[4], embed_fn_saving=True)
    sys.exit(0);
# }}}

# {{{ nsf5
def nsf5():

    if len(sys.argv)!=5:
        print(sys.argv[0], "nsf5-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("NSF5")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.nsf5, 
            sys.argv[2], sys.argv[3], sys.argv[4], embed_fn_saving=True)
    sys.exit(0);
# }}}

# {{{ nsf5_color
def nsf5_color():

    if len(sys.argv)!=5:
        print(sys.argv[0], "nsf5-color-sim <image/dir> <payload> <output-dir>\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("NSF5_COLOR")

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.nsf5_color, 
            sys.argv[2], sys.argv[3], sys.argv[4], embed_fn_saving=True)
    sys.exit(0);
# }}}

# {{{ experimental
def experimental():

    if len(sys.argv)!=5:
        print(sys.argv[0], "experimental-sim <image/dir> <payload> <output-dir>")
        print("NOTE: Please, put your EXPERIMENTAL.m file into external/octave\n")
        sys.exit(0)

    import aletheialib.stegosim
    aletheialib.stegosim.embed_message(aletheialib.stegosim.experimental, sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0);
# }}}

# {{{ steghide
def steghide():

    if len(sys.argv)!=5:
        print(sys.argv[0], "steghide-sim <image/dir> <payload> <output-dir>")
        sys.exit(0)

    import aletheialib.stegosim
    import aletheialib.utils
    aletheialib.utils.check_bin("steghide")

    aletheialib.stegosim.embed_message(aletheialib.stegosim.steghide, sys.argv[2], sys.argv[3], sys.argv[4],
                  embed_fn_saving=True)
    sys.exit(0);
# }}}

# {{{ outguess
def outguess():

    if len(sys.argv)!=5:
        print(sys.argv[0], "outguess-sim <image/dir> <payload> <output-dir>")
        sys.exit(0)

    import aletheialib.stegosim
    import aletheialib.utils
    aletheialib.utils.check_bin("outguess")

    aletheialib.stegosim.embed_message(aletheialib.stegosim.outguess, sys.argv[2], sys.argv[3], sys.argv[4],
                  embed_fn_saving=True)
    sys.exit(0);
# }}}

# {{{ steganogan
def steganogan():

    if len(sys.argv)!=5:
        print(sys.argv[0], "steganogan-sim <image/dir> <payload> <output-dir>")
        sys.exit(0)

    import aletheialib.stegosim
    import aletheialib.utils
    aletheialib.utils.check_bin("steghide")

    aletheialib.stegosim.embed_message(aletheialib.stegosim.steganogan, 
            sys.argv[2], sys.argv[3], sys.argv[4], embed_fn_saving=True)
    sys.exit(0);
# }}}












