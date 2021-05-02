import sys

doc="\n" \
"  Feature extractors:\n" \
"  - srm:           Full Spatial Rich Models.\n" \
"  - srmq1:         Spatial Rich Models with fixed quantization q=1c.\n" \
"  - scrmq1:        Spatial Color Rich Models with fixed quantization q=1c.\n" \
"  - gfr:           JPEG steganalysis with 2D Gabor Filters.\n" \
"  - hill-maxsrm:   Selection-Channel-Aware Spatial Rich Models for HILL." 


# {{{ srm
def srm():
    if len(sys.argv)!=4:
        print(sys.argv[0], "srm <image/dir> <output-file>\n")
        sys.exit(0)

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

# {{{ hill-sigma-spam-psrm
def hill_sigma_spam_psrm():

    if len(sys.argv)!=4:
        print(sys.argv[0], "hill-sigma-spam-psrm <image/dir> <output-file>\n")
        sys.exit(0)

    import aletheialib.feaext

    image_path=sys.argv[2]
    ofile=sys.argv[3]

    aletheialib.feaext.extract_features(aletheialib.feaext.HILL_sigma_spam_PSRM_extract, image_path, ofile)
    sys.exit(0)
# }}}

# {{{ hill-maxsrm
def hill_maxsrm():

    if len(sys.argv)!=4:
        print(sys.argv[0], "hill-maxsrm <image/dir> <output-file>\n")
        sys.exit(0)

    import aletheialib.feaext

    image_path=sys.argv[2]
    ofile=sys.argv[3]

    aletheialib.feaext.extract_features(aletheialib.feaext.HILL_MAXSRM_extract, image_path, ofile)
    sys.exit(0)
# }}}




