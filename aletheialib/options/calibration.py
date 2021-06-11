
import os
import sys
import glob

doc = "\n" \
"  Calibration attacks to JPEG steganography:\n" \
"  - calibration:   Calibration attack on F5."

def launch():
    if len(sys.argv)!=3:
        #print(sys.argv[0], "calibration <f5|chisquare_mode> <image>\n")
        print(sys.argv[0], "calibration <image|dir>\n")
        sys.exit(0)

    #if sys.argv[2] not in ["f5", "chisquare_mode"]:
    #    print("Please, provide a valid method")
    #    sys.exit(0)


    import aletheialib.utils
    import aletheialib.attacks


    if os.path.isdir(sys.argv[2]):
        for f in glob.glob(os.path.join(sys.argv[2], "*")):
            if not aletheialib.utils.is_valid_image(f):
                print(f, " is not a valid image")
                continue
     
            if not f.lower().endswith(('.jpg', '.jpeg')):
                print(f, "is not a JPEG file")
                continue

            try:
                fn = aletheialib.utils.absolute_path(f)
                beta = aletheialib.attacks.calibration_f5_octave_jpeg(fn, True)
                print(f, ", beta:", beta)
            except:
                print("Error processing", f)
    else:

        if not aletheialib.utils.is_valid_image(sys.argv[2]):
            print("Please, provide a valid image")
            sys.exit(0)

        if not sys.argv[2].lower().endswith(('.jpg', '.jpeg')):
            print("Please, provide a JPEG file")
            sys.exit(0)

        fn = aletheialib.utils.absolute_path(sys.argv[2])
        aletheialib.attacks.calibration_f5_octave_jpeg(fn)
        #print(aletheialib.attacks.calibration_f5_octave_jpeg(fn, True))

        #if "f5" in sys.argv[2]:
        #    aletheialib.attacks.calibration_f5(fn)
        #elif "chisquare_mode" in sys.argv[2]:
        #    aletheialib.attacks.calibration_chisquare_mode(fn)





