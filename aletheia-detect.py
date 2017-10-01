#!/usr/bin/python


import sys
import json
from aletheia import attacks, imutils
#from cnn import net as cnn

def main():

    if len(sys.argv)!=3:
        print sys.argv[0], "<command> <image>\n"
        print "Commands: "
        print "  exif:    Show EXIF metadata."
        print "  spa:     Sample Pairs Analysis attack to LSB replacement."
        print "  rs:      RS attack to LSB replacement."
        print "\n"
        sys.exit(0)


    # {{{ spa
    if sys.argv[1]=="spa":
   
        if not imutils.is_valid_image(sys.argv[2]):
            print "Please, provide a valid image"
            sys.exit(0)

        threshold=0.05
        bitrate_R=attacks.spa(sys.argv[2], 0)
        bitrate_G=attacks.spa(sys.argv[2], 1)
        bitrate_B=attacks.spa(sys.argv[2], 2)

        if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
            print "No hiden data found"
            sys.exit(0)

        if bitrate_R>=threshold:
            print "Hiden data found in channel R", bitrate_R
        if bitrate_G>=threshold:
            print "Hiden data found in channel G", bitrate_G
        if bitrate_B>=threshold:
            print "Hiden data found in channel B", bitrate_B
        sys.exit(0)
    # }}}

    # {{{ rs
    if sys.argv[1]=="rs":

        if not imutils.is_valid_image(sys.argv[2]):
            print "Please, provide a valid image"
            sys.exit(0)

        threshold=0.05
        bitrate_R=attacks.rs(sys.argv[2], 0)
        bitrate_G=attacks.rs(sys.argv[2], 1)
        bitrate_B=attacks.rs(sys.argv[2], 2)

        if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
            print "No hiden data found"
            sys.exit(0)

        if bitrate_R>=threshold:
            print "Hiden data found in channel R", bitrate_R
        if bitrate_G>=threshold:
            print "Hiden data found in channel G", bitrate_G
        if bitrate_B>=threshold:
            print "Hiden data found in channel B", bitrate_B
        sys.exit(0)
    # }}}



if __name__ == "__main__":
    main()



