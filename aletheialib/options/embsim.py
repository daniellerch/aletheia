import sys
import os
import urllib.request
import urllib.error

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


EXTERNAL_RESOURCES='https://raw.githubusercontent.com/daniellerch/aletheia-external-resources/main/'


# {{{ download_octave_code()
def download_octave_code(method):

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir, os.pardir))
    cache_dir = os.path.join(basedir, 'aletheia-cache', 'octave')

    remote_octave_file = EXTERNAL_RESOURCES+'octave/code/'+method+'.m'
    remote_license_file = EXTERNAL_RESOURCES+'octave/code/'+method+'.LICENSE'

    local_octave_file = os.path.join(cache_dir, method+'.m')
    local_license_file = os.path.join(cache_dir, method+'.LICENSE')

    # Has the file already been downloaded?
    if os.path.isfile(local_octave_file): 
        return

    # Download the license if available
    try:
        urllib.request.urlretrieve(remote_license_file, local_license_file)
    except:
        print("Error,", method, "license cannot be downloaded")
        sys.exit(0)


    # Accept license
    print("\nCODE:", method)
    print("\nTo proceed, kindly confirm your acceptance of the license and your")
    print("agreement to download the code from 'aletheia-external-resources'\n")

    print("LICENSE:\n")
    with open(local_license_file, 'r') as f:
        print(f.read())

    r = ""
    while r not in ["y", "n"]:
        r = input("Do you agree? (y/n): ")
        if r == "n":
            print("The terms have not been accepted\n")
            sys.exit(0)
        elif r == "y":
            break

    # Download code
    try:
        urllib.request.urlretrieve(remote_octave_file, local_octave_file)
    except:
        print("Error,", method, "code cannot be downloaded")
        sys.exit(0)

# }}}

# {{{ download_octave_jpeg_toolbox()
def download_octave_jpeg_toolbox():

    currdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(currdir, os.pardir, os.pardir))
    cache_dir = os.path.join(basedir, 'aletheia-cache')

    # Has the JPEG TOOLBOX already been downloaded?
    if os.path.isfile(os.path.join(cache_dir, 'jpeg_toolbox/Makefile')):
        return

    # Download the license if available
    os.makedirs(os.path.join(cache_dir, "jpeg_toolbox"), exist_ok=True)
    licpath = 'jpeg_toolbox/LICENSE'
    remote_license_file = EXTERNAL_RESOURCES+'octave/'+licpath
    local_license_file = os.path.join(cache_dir, licpath)
    try:
        urllib.request.urlretrieve(remote_license_file, local_license_file)
    except Exception as e:
        print("Error, JPEG TOOLBOX license cannot be downloaded")
        sys.exit(0)

    # Accept license
    print("\nTo use this command, you must accept the license of the JPEG TOOLBOX. In that")
    print("case, we will download the JPEG TOOLBOX from 'aletheia-external-resources'.\n")
    with open(local_license_file, 'r') as f:
        print(f.read())

    r = ""
    while r not in ["y", "n"]:
        r = input("Do you accept the license? (y/n): ")
        if r == "n":
            print("The license has not been accepted\n")
            sys.exit(0)
        elif r == "y":
            break

    # Download JPEG TOOLBOX
    for f in ['jpeg_read.c', 'jpeg_write.c', 'Makefile']:
        remote_file = EXTERNAL_RESOURCES+'octave/jpeg_toolbox/'+f
        local_file = os.path.join(cache_dir, 'jpeg_toolbox/'+f)
        try:
            urllib.request.urlretrieve(remote_file, local_file)
        except:
            print("Error,", remote_file, "cannot be downloaded")
            sys.exit(0)






    
# }}}


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












