import sys

doc="\n" \
"  Embedding simulators:\n" \
"  - lsbr-sim:             Embedding using LSB replacement simulator.\n" \
"  - lsbm-sim:             Embedding using LSB matching simulator.\n" \
"  - hugo-sim:             Embedding using HUGO simulator.\n" \
"  - wow-sim:              Embedding using WOW simulator.\n" \
"  - s-uniward-sim:        Embedding using S-UNIWARD simulator.\n" \
"  - s-uniward-color-sim:  Embedding using S-UNIWARD color simulator.\n" \
"  - j-uniward-sim:        Embedding using J-UNIWARD simulator.\n" \
"  - j-uniward-color-sim:  Embedding using J-UNIWARD color simulator.\n" \
"  - hill-sim:             Embedding using HILL simulator.\n" \
"  - hill-color-sim:       Embedding using HILL color simulator.\n" \
"  - ebs-sim:              Embedding using EBS simulator.\n" \
"  - ebs-color-sim:        Embedding using EBS color simulator.\n" \
"  - ued-sim:              Embedding using UED simulator.\n" \
"  - ued-color-sim:        Embedding using UED color simulator.\n" \
"  - nsf5-sim:             Embedding using nsF5 simulator.\n" \
"  - nsf5-color-sim:       Embedding using nsF5 color simulator.\n" \
"  - steghide-sim:         Embedding using Steghide simulator.\n" \
"  - outguess-sim:         Embedding using Outguess simulator.\n" \
"  - steganogan-sim:       Embedding using SteganoGAN simulator." 




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












