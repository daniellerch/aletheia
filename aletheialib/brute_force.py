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


# {{{ check_password()
def check_password(params):
    passw, command, use_filetype, continue_searching = params
    cmd = command.replace("<PASSWORD>", passw)

    if use_filetype:
        tempd = tempfile.mkdtemp()
        tempf = os.path.join(tempd, 'output')
        cmd = cmd.replace("<OUTPUT_FILE>", tempf)

    FNUL = open(os.devnull, 'w')
    p=subprocess.Popen(cmd, stdout=FNUL, stderr=FNUL, shell=True)
    #output, err = p.communicate()
    status = p.wait()

    if use_filetype:
        ft = None
        try:
            ft = magic.Magic(mime=True).from_file(tempf)
        except:
            pass
        if ft != None and ft != 'inode/x-empty' and ft != 'application/octet-stream':
            if not continue_searching:
                print("\nPassword found:", passw)
                shutil.rmtree(tempd, ignore_errors=True)
                return True

            print(f"Candidate password: {passw}, filetype found: {ft}")

            """
            if ft == "text/plain":
                with open(tempf) as f:
                    try:
                        s = f.read().decode('ascii')
                        if s.isprintable():
                            print(f"Candidate password: {passw}, filetype found: {ft}")
                            print(s)
                    except:
                        pass
            """
        shutil.rmtree(tempd, ignore_errors=True)

    elif p.returncode==0:
        print("\nPassword found:", passw)
        return True
    return False
# }}}

# {{{ generic()
def generic(command, password_file, use_filetype=False, continue_searching=False):
    
    with open(password_file, "rU") as f:
        passwords = f.readlines()

    params = [ (passwd.replace("\n", ""), 
                command, 
                use_filetype, 
                continue_searching) for passwd in passwords ]

    n_proc = cpu_count()
    print("Using", n_proc, "processes")
    pool = ThreadPool(n_proc)

    # Process thread pool in batches
    batch=100
    for i in range(0, len(params), batch):
        perc = round(100*float(i)/len(passwords),2)
        sys.stdout.write("Completed: "+str(perc)+'%    \r')
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
    aletheialib.utils.check_bin("outguess")   
    command = f"outguess -k <PASSWORD> -r {path} <OUTPUT_FILE>"
    generic(command, password_file, use_filetype=True, continue_searching=True)

# }}}

# {{{ openstego()
def openstego(path, password_file):
    aletheialib.utils.check_bin("openstego")   
    command = f"openstego extract -sf stego.png -p <PASSWORD> -xf <OUTPUT_FILE>"
    generic(command, password_file, use_filetype=True, continue_searching=False)

# }}}



