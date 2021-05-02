
import sys

doc = "\n" \
"  Find password by brute force using a list of passwords:\n" \
"  - bf-generic:   Generic tool for finding the password using a command tool."


# {{{ generic()
def generic():

    if len(sys.argv)!=4:
        print(sys.argv[0], "bf-generic <unhide command> <passw file>\n")
        print("Example:")
        print(sys.argv[0], "bf-generic 'steghide extract -sf image.jpg -xf output.txt -p <PASSWORD> -f' resources/passwords.txt\n")
        print("")
        sys.exit(0)

    import aletheialib.attacks
    aletheialib.attacks.brute_force(sys.argv[2], sys.argv[3])
# }}}


