#!/usr/bin/python

import os
import sys
import magic
import shutil
import tempfile
import subprocess
import aletheialib.utils

from multiprocessing import Pool as ThreadPool 
from multiprocessing import cpu_count

IGNORE_FILETYPES = ['inode/x-empty', 'application/x-dosexec', 'application/octet-stream']


# {{{ check_password()
def check_password(params):
    passw, command, use_filetype, continue_searching, success_output_string = params
    cmd = command.replace("<PASSWORD>", passw)

    if use_filetype:
        tempd = tempfile.mkdtemp()
        tempf = os.path.join(tempd, 'output')
        cmd = cmd.replace("<OUTPUT_FILE>", tempf)


    FNUL = open(os.devnull, 'w')
    p=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=FNUL, shell=True)
    output, err = p.communicate()
    status = p.wait()


    if success_output_string != None:
        for line in output.decode().split('\n'):
            if success_output_string in line:
                print("\nPassword found:", passw)
                print(line)
                return True
        return False


    if use_filetype:
        ft = None
        try:
            ft = magic.Magic(mime=True).from_file(tempf)
        except:
            pass
        if ft != None and ft not in IGNORE_FILETYPES:
            if not continue_searching:
                print("\nPassword found:", passw)
                shutil.rmtree(tempd, ignore_errors=True)
                return True

            content = ""
            candidate = True
            if ft == "text/plain":
                encoding = magic.Magic(mime_encoding=True).from_file(tempf)
                if encoding != "unknown-8bit":
                    with open(tempf, 'rb') as f:
                        try:
                            s = f.read().decode(encoding)
                            content = ', content: "'+s[:32]+'..." ('+encoding+')'
                        except:
                            candidate = False
                            pass
                else:
                    candidate = False

            if candidate:
                size = os.path.getsize(tempf)
                print(f"Candidate password: {passw}, ft: {ft}, size: {size} bytes{content}")

        shutil.rmtree(tempd, ignore_errors=True)

    elif p.returncode==0:
        print("\nPassword found:", passw)
        return True
    return False
# }}}

# {{{ generic()
def generic(command, password_file, use_filetype=False, 
            continue_searching=False, success_output_string=None):
    
    if not os.path.isfile(password_file):
        print("ERROR: File not found -", password_file)
        sys.exit(0)

    with open(password_file, "rU") as f:
        passwords = f.readlines()

    params = [ (passwd.replace("\n", ""), 
                command, 
                use_filetype, 
                continue_searching,
                success_output_string) for passwd in passwords ]

    n_proc = cpu_count()
    print("Using", n_proc, "processes")
    pool = ThreadPool(n_proc)

    # Process thread pool in batches
    batch=100
    for i in range(0, len(params), batch):
        perc = round(100*float(i)/len(passwords),2)
        sys.stderr.write("Completed: "+str(perc)+'%    \r')
        pool = ThreadPool(n_proc)
        results = pool.map(check_password, params[i:i+batch])
        pool.close()
        pool.terminate()
        pool.join()
        if any(results):
            break

# }}}

# {{{ steghide()
def steghide(path, password_file):

    if not os.path.isfile(password_file):
        print("ERROR: File not found -", password_file)
        sys.exit(0)

    aletheialib.utils.check_bin("steghide")   
    command = f"steghide extract -sf {path} -xf output.txt -p <PASSWORD> -f"
    generic(command, password_file, use_filetype=False, continue_searching=False)

# }}}

# {{{ outguess()
def outguess(path, password_file):
    """
    Outguess does not returns a right code when the password is OK, so we 
    need to check the extracted file type to know if the password is a good
    candidate.
    """

    if not os.path.isfile(password_file):
        print("ERROR: File not found -", password_file)
        sys.exit(0)

    aletheialib.utils.check_bin("outguess")   
    command = f"outguess -k <PASSWORD> -r {path} <OUTPUT_FILE>"
    generic(command, password_file, use_filetype=True, continue_searching=True)

# }}}

# {{{ openstego()
def openstego(path, password_file):

    if not os.path.isfile(password_file):
        print("ERROR: File not found -", password_file)
        sys.exit(0)

    aletheialib.utils.check_bin("openstego")   
    command = f"openstego extract -sf {path} -p <PASSWORD> -xf <OUTPUT_FILE>"
    generic(command, password_file, use_filetype=True, continue_searching=False)

# }}}

# {{{ f5()
def f5(path, password_file):
    """
    F5 does not returns a right code when the password is OK, so we 
    need to check the extracted file type to know if the password is a good
    candidate.
    """
    
    if not os.path.isfile(password_file):
        print("ERROR: File not found -", password_file)
        sys.exit(0)

    # Password path
    if os.path.isabs(password_file):
        pass_path = password_file
    else:
        pass_path = os.path.join(os.getcwd(), password_file)

    # Image path
    if os.path.isabs(path):
        image_path = path
    else:
        image_path = os.path.join(os.getcwd(), path)


    # Get the directory where the resources are installed
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, os.pardir, 'aletheia-cache', 'F5')

    os.chdir(dir_path)

    aletheialib.utils.check_bin("java")   

    command = f"java Extract -p <PASSWORD> -e <OUTPUT_FILE> {image_path} "
    generic(command, pass_path, use_filetype=True, continue_searching=True)

# }}}

# {{{ stegosuite()
def stegosuite(path, password_file):

    if not os.path.isfile(password_file):
        print("ERROR: File not found -", password_file)
        sys.exit(0)

    aletheialib.utils.check_bin("stegosuite")   
    command = f"stegosuite -k <PASSWORD> -x {path} "
    generic(command, password_file, use_filetype=False, 
            continue_searching=False, 
            success_output_string="Extracted message")

# }}}

