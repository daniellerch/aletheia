
import sys

doc = "\n" \
"  Calibration attacks to JPEG steganography:\n" \
"  - calibration:   Calibration attack to JPEG images."

def launch():
    if len(sys.argv)!=4:
        print(sys.argv[0], "calibration <f5|chisquare_mode> <image>\n")
        sys.exit(0)

    if sys.argv[2] not in ["f5", "chisquare_mode"]:
        print("Please, provide a valid method")
        sys.exit(0)


    import aletheialib.utils
    import aletheialib.attacks

    if not aletheialib.utils.is_valid_image(sys.argv[3]):
        print("Please, provide a valid image")
        sys.exit(0)

    if not sys.argv[3].lower().endswith(('.jpg', '.jpeg')):
        print("Please, provide a JPEG file")
        sys.exit(0)

    fn = aletheialib.utils.absolute_path(sys.argv[3])
    if "f5" in sys.argv[2]:
        aletheialib.attacks.calibration_f5(fn)
    elif "chisquare_mode" in sys.argv[2]:
        aletheialib.attacks.calibration_chisquare_mode(fn)





