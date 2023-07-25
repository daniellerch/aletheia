import sys
from aletheialib.utils import download_octave_code, download_octave_jpeg_toolbox

doc="\n" \
"  Feature extractors:\n" \
"  - srm:           Full Spatial Rich Models.\n" \
"  - srmq1:         Spatial Rich Models with fixed quantization q=1c.\n" \
"  - scrmq1:        Spatial Color Rich Models with fixed quantization q=1c.\n" \
"  - gfr:           JPEG steganalysis with 2D Gabor Filters.\n" \
"  - dctr:          JPEG Low complexity features extracted from DCT residuals.\n" \


# {{{ srm
def srm():
    if len(sys.argv)!=4:
        print(sys.argv[0], "srm <image/dir> <output-file>\n")
        sys.exit(0)

    download_octave_code("SRM")

    import aletheialib.feaext

    image_path=sys.argv[2]
    ofile=sys.argv[3]

    aletheialib.feaext.extract_features(aletheialib.feaext.SRM_extract, image_path, ofile)
    sys.exit(0)
# }}}

# {{{ srmq1
def srmq1():
    if len(sys.argv)!=4:
        print(sys.argv[0], "srmq1 <image/dir> <output-file>\n")
        sys.exit(0)

    download_octave_code("SRMQ1")

    import aletheialib.feaext

    image_path=sys.argv[2]
    ofile=sys.argv[3]

    aletheialib.feaext.extract_features(aletheialib.feaext.SRMQ1_extract, image_path, ofile)
    sys.exit(0)
# }}}

# {{{ scrmq1
def scrmq1():

    if len(sys.argv)!=4:
        print(sys.argv[0], "scrmq1 <image/dir> <output-file>\n")
        sys.exit(0)

    download_octave_code("SCRMQ1")

    import aletheialib.feaext

    image_path=sys.argv[2]
    ofile=sys.argv[3]

    aletheialib.feaext.extract_features(aletheialib.feaext.SCRMQ1_extract, image_path, ofile)
    sys.exit(0)
# }}}

# {{{ gfr
def gfr():

    if len(sys.argv)<4:
        print(sys.argv[0], "gfr <image/dir> <output-file> [quality] [rotations]\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("GFR")

    import aletheialib.feaext

    image_path=sys.argv[2]
    ofile=sys.argv[3]

    if len(sys.argv)<5:
        quality = "auto"
        print("JPEG quality not provided, using detection via 'identify'")
    else:
        quality = sys.argv[4]


    if len(sys.argv)<6:
        rotations = 32
        print("Number of rotations for Gabor kernel no provided, using:", \
              rotations)
    else:
        rotations = sys.argv[6]


    params = {
        "quality": quality,
        "rotations": rotations
    }
        
    aletheialib.feaext.extract_features(aletheialib.feaext.GFR_extract, image_path, ofile, params)
    sys.exit(0)
# }}}

# {{{ dctr
def dctr():

    if len(sys.argv)<4:
        print(sys.argv[0], "dctr <image/dir> <output-file> [quality]\n")
        sys.exit(0)

    download_octave_jpeg_toolbox()
    download_octave_code("DCTR")

    import aletheialib.feaext

    image_path=sys.argv[2]
    ofile=sys.argv[3]

    if len(sys.argv)<5:
        quality = "auto"
        print("JPEG quality not provided, using detection via 'identify'")
    else:
        quality = sys.argv[4]



    params = {
        "quality": quality,
    }
        
    aletheialib.feaext.extract_features(aletheialib.feaext.DCTR_extract, image_path, ofile, params)
    sys.exit(0)
# }}}





