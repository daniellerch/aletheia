
import os
import glob
import sys

import numpy as np
import aletheialib.utils
import aletheialib.attacks
from aletheialib.utils import download_octave_code

from imageio import imread
from multiprocessing import Pool as ThreadPool 
from multiprocessing import cpu_count

from PIL import Image
import aletheialib.octave_interface as O



doc = "\n" \
"  Structural LSB detectors (Statistical attacks to LSB replacement):\n" \
"  - spa:           Sample Pairs Analysis.\n" \
"  - rs:            RS attack.\n" \
"  - ws:            Weighted Stego Attack.\n" \
"  - triples:       Triples Attack.\n"



# {{{ spa()
def spa():
    if len(sys.argv)<3:
        print(sys.argv[0], "spa <image> [threshold]\n")
        sys.exit(0)

    import aletheialib.utils
    import aletheialib.attacks
    from imageio import imread

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)

    threshold = 0.05
    if len(sys.argv) == 4:
        threshold = float(sys.argv[3])
    print("Using threshold:", threshold)

    I = imread(sys.argv[2])
    if len(I.shape)==2:
        bitrate = aletheialib.attacks.spa_image(I, None)
        if bitrate<threshold:
            print("No hidden data found")
        else:
            print("Hidden data found"), bitrate
    else:
        bitrate_R = aletheialib.attacks.spa_image(I, 0)
        bitrate_G = aletheialib.attacks.spa_image(I, 1)
        bitrate_B = aletheialib.attacks.spa_image(I, 2)

        if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
            print("No hidden data found")
            sys.exit(0)

        if bitrate_R>=threshold:
            print("Hidden data found in channel R", bitrate_R)
        if bitrate_G>=threshold:
            print("Hidden data found in channel G", bitrate_G)
        if bitrate_B>=threshold:
            print("Hidden data found in channel B", bitrate_B)
    sys.exit(0)
# }}}

# {{{ rs()
def rs():
    if len(sys.argv)<3:
        print(sys.argv[0], "rs <image> [threshold]\n")
        sys.exit(0)

    import numpy as np
    import aletheialib.utils
    import aletheialib.attacks
    from imageio import imread

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)

    threshold=0.05
    if len(sys.argv) == 4:
        threshold = float(sys.argv[3])
    print("Using threshold:", threshold)

    I = np.asarray(imread(sys.argv[2]))
    if len(I.shape)==2:
        bitrate = aletheialib.attacks.rs_image(I)
        if bitrate<threshold:
            print("No hidden data found")
        else:
            print("Hidden data found", bitrate)
    else:
        bitrate_R = aletheialib.attacks.rs_image(I, 0)
        bitrate_G = aletheialib.attacks.rs_image(I, 1)
        bitrate_B = aletheialib.attacks.rs_image(I, 2)

        if bitrate_R<threshold and bitrate_G<threshold and bitrate_B<threshold:
            print("No hidden data found")
            sys.exit(0)

        if bitrate_R>=threshold:
            print("Hidden data found in channel R", bitrate_R)
        if bitrate_G>=threshold:
            print("Hidden data found in channel G", bitrate_G)
        if bitrate_B>=threshold:
            print("Hidden data found in channel B", bitrate_B)
        sys.exit(0)
# }}}

# {{{ ws()
def ws(): 
    if len(sys.argv)<3:
        print(sys.argv[0], "ws <image> [threshold]\n")
        sys.exit(0)

    download_octave_code("WS")

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)

    threshold=0.05
    if len(sys.argv) == 4:
        threshold = float(sys.argv[3])
    print("Using threshold:", threshold)

    path = aletheialib.utils.absolute_path(sys.argv[2])
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('WS', path, params={"channel":1})["data"][0][0]
        alpha_G = O._attack('WS', path, params={"channel":2})["data"][0][0]
        alpha_B = O._attack('WS', path, params={"channel":3})["data"][0][0]

        if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
            print("No hidden data found")

        if alpha_R>=threshold:
            print("Hidden data found in channel R", alpha_R)
        if alpha_G>=threshold:
            print("Hidden data found in channel G", alpha_G)
        if alpha_B>=threshold:
            print("Hidden data found in channel B", alpha_B)

    else:
        alpha = O._attack('WS', path, params={"channel":1})["data"][0][0]
        if alpha>=threshold:
            print("Hidden data found", alpha)
        else:
            print("No hidden data found")

    sys.exit(0)
# }}}

# {{{ triples()
def triples():

    if len(sys.argv)<3:
        print(sys.argv[0], "triples <image> [threshold]\n")
        sys.exit(0)

    download_octave_code("TRIPLES")

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)


    threshold=0.05
    if len(sys.argv) == 4:
        threshold = float(sys.argv[3])
    print("Using threshold:", threshold)

    path = aletheialib.utils.absolute_path(sys.argv[2])
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('TRIPLES', path, params={"channel":1})["data"][0][0]
        alpha_G = O._attack('TRIPLES', path, params={"channel":2})["data"][0][0]
        alpha_B = O._attack('TRIPLES', path, params={"channel":3})["data"][0][0]


        if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
            print("No hidden data found")

        if alpha_R>=threshold:
            print("Hidden data found in channel R", alpha_R)
        if alpha_G>=threshold:
            print("Hidden data found in channel G", alpha_G)
        if alpha_B>=threshold:
            print("Hidden data found in channel B", alpha_B)

    else:
        alpha = O._attack('TRIPLES', path, params={"channel":1})["data"][0][0]
        if alpha>=threshold:
            print("Hidden data found", alpha)
        else:
            print("No hidden data found")

    sys.exit(0)
# }}}

# {{{ aump()
def aump():
    if len(sys.argv)<3:
        print(sys.argv[0], "aump <image> [threshold]\n")
        sys.exit(0)

    download_octave_code("AUMP")

    if not aletheialib.utils.is_valid_image(sys.argv[2]):
        print("Please, provide a valid image")
        sys.exit(0)


    threshold=0.05
    if len(sys.argv) == 4:
        threshold = float(sys.argv[3])
    print("Using threshold:", threshold)

    path = aletheialib.utils.absolute_path(sys.argv[2])
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('AUMP', path, params={"channel":1})["data"][0][0]
        alpha_G = O._attack('AUMP', path, params={"channel":2})["data"][0][0]
        alpha_B = O._attack('AUMP', path, params={"channel":3})["data"][0][0]

        if alpha_R<threshold and alpha_G<threshold and alpha_B<threshold:
            print("No hidden data found")

        if alpha_R>=threshold:
            print("Hidden data found in channel R", alpha_R)
        if alpha_G>=threshold:
            print("Hidden data found in channel G", alpha_G)
        if alpha_B>=threshold:
            print("Hidden data found in channel B", alpha_B)
    else:
        alpha = O._attack('WS', path, params={"channel":1})["data"][0][0]
        if alpha>=threshold:
            print("Hidden data found", alpha)
        else:
            print("No hidden data found")


    sys.exit(0)
# }}}


# {{{ spa_score
def spa_detect(p):
    f, threshold = p
    I = imread(f)
    if len(I.shape)==2:
        bitrate = aletheialib.attacks.spa_image(I, None)
    else:
        bitrate_R = aletheialib.attacks.spa_image(I, 0)
        bitrate_G = aletheialib.attacks.spa_image(I, 1)
        bitrate_B = aletheialib.attacks.spa_image(I, 2)
        bitrate = max(bitrate_R, bitrate_G, bitrate_B)

    if bitrate < threshold:
        return False
    return True

def spa_score():

    if len(sys.argv)<4:
        print(sys.argv[0], "spa-score <test-cover-dir> <test-stego-dir> [threshold]\n")
        print("     test-cover-dir:    Directory containing cover images")
        print("     test-stego-dir:    Directory containing stego images")
        print("     threshold:         Threshold for detecting steganographic images (default=0.05)")
        print("")
        sys.exit(0)

    cover_dir=sys.argv[2]
    stego_dir=sys.argv[3]

    cover_files = sorted(glob.glob(os.path.join(cover_dir, '*')))
    stego_files = sorted(glob.glob(os.path.join(stego_dir, '*')))

    if len(sys.argv)==5:
        threshold = float(sys.argv[4])
    else:
        threshold = 0.05
    print("Using threshold", threshold)

    batch=1000
    n_core = cpu_count()

    # Process thread pool in batches
    for i in range(0, len(cover_files), batch):
        params_batch = zip(cover_files[i:i+batch], [threshold]*batch)
        pool = ThreadPool(n_core)
        pred_cover = pool.map(spa_detect, params_batch)
        pool.close()
        pool.terminate()
        pool.join()

    for i in range(0, len(stego_files), batch):
        params_batch = zip(stego_files[i:i+batch], [threshold]*batch)
        pool = ThreadPool(n_core)
        pred_stego = pool.map(spa_detect, params_batch)
        pool.close()
        pool.terminate()
        pool.join()

    ok = np.sum(np.array(pred_cover)==0)+np.sum(np.array(pred_stego)==1)
    score = ok/(len(pred_cover)+len(pred_stego))
    print("score:", score)

# }}}

# {{{ ws_score
def ws_detect(f):

    if not aletheialib.utils.is_valid_image(f):
        print("Please, provide a valid image:", f)
        return False

    threshold=0.05
    path = aletheialib.utils.absolute_path(f)
    im=Image.open(path)
    if im.mode in ['RGB', 'RGBA', 'RGBX']:
        alpha_R = O._attack('WS', path, params={"channel":1})["data"][0][0]
        alpha_G = O._attack('WS', path, params={"channel":2})["data"][0][0]
        alpha_B = O._attack('WS', path, params={"channel":3})["data"][0][0]
        alpha = max(alpha_R, alpha_G, alpha_B)
    else:
        alpha = O._attack('WS', path, params={"channel":1})["data"][0][0]

    if alpha<threshold:
        return False
    return True


def ws_score():

    if len(sys.argv)<3:
        print(sys.argv[0], "ws-score <test-cover-dir> <test-stego-dir>\n")
        print("     test-cover-dir:    Directory containing cover images")
        print("     test-stego-dir:    Directory containing stego images")
        print("")
        sys.exit(0)

    cover_dir=sys.argv[2]
    stego_dir=sys.argv[3]

    cover_files = sorted(glob.glob(os.path.join(cover_dir, '*')))
    stego_files = sorted(glob.glob(os.path.join(stego_dir, '*')))


    batch=1000
    n_core = cpu_count()

    # Process thread pool in batches
    for i in range(0, len(cover_files), batch):
        params_batch = cover_files[i:i+batch]
        pool = ThreadPool(n_core)
        pred_cover = pool.map(ws_detect, params_batch)
        pool.close()
        pool.terminate()
        pool.join()

    for i in range(0, len(stego_files), batch):
        params_batch = stego_files[i:i+batch]
        pool = ThreadPool(n_core)
        pred_stego = pool.map(ws_detect, params_batch)
        pool.close()
        pool.terminate()
        pool.join()

    ok = np.sum(np.array(pred_cover)==0)+np.sum(np.array(pred_stego)==1)
    score = ok/(len(pred_cover)+len(pred_stego))
    print("score:", score)

# }}}





