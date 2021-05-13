
import sys

doc = "\n" \
"  Find password by brute force using a list of passwords:\n" \
"  - brute-force-steghide:  Brute force a password using StegHide"
"  - brute-force-generic:   Generic tool for finding the password using a command"


# {{{ generic()
def generic():

    if len(sys.argv)!=4:
        print(sys.argv[0], "brute-force-generic <unhide command> <passw file>\n")
        print("Example:")
        print(sys.argv[0], "brute-force-generic 'steghide extract -sf image.jpg -xf output.txt -p <PASSWORD> -f' resources/passwords.txt\n")
        print("")
        sys.exit(0)

    import aletheialib.brute_force
    aletheialib.brute_force.generic(sys.argv[2], sys.argv[3])
# }}}

# {{{ steghide()
def steghide():

    if len(sys.argv)!=4:
        print(sys.argv[0], "brute-force-steghide <image> <passw file>\n")
        print("Example:")
        print(sys.argv[0], "brute-force-steghide image.jpg resources/passwords.txt\n")
        print("")
        sys.exit(0)

    import aletheialib.brute_force
    aletheialib.brute_force.steghide(sys.argv[2], sys.argv[3])
# }}}

# {{{ outguess()
def outguess():

    if len(sys.argv)!=4:
        print(sys.argv[0], "brute-force-outguess <image> <passw file>\n")
        print("Example:")
        print(sys.argv[0], "brute-force-outguess image.jpg resources/passwords.txt\n")
        print("")
        sys.exit(0)

    import aletheialib.brute_force
    aletheialib.brute_force.outguess(sys.argv[2], sys.argv[3])
# }}}

