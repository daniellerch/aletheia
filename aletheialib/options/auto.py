
import os
import sys
import glob
import aletheialib.utils
import numpy as np

from aletheialib.utils import download_octave_code
from aletheialib.utils import download_octave_jpeg_toolbox


doc = "\n" \
"  Automated tools:\n" \
"  - auto:      Try different steganalysis methods.\n" \
"  - dci:       Predicts a set of images using DCI evaluation.\n" \
"  - dci-si:    Predict an image using the DCI-SI method."



def _format_line(value, length, threshold=0.5):
    # {{{
    if value==1:
        value = 0.9
    if value > threshold:
        return ("["+str(round(value,1))+"]").center(length, ' ')
    return str(round(value,1)).center(length, ' ')
    # }}}

def _format_line_2(value1, value2, length):
    # {{{
    if value1==1:
        value1 = 0.9
    v1 = str(round(value1,1))
    v2 = str(round(value2,1))
    if value1 > 0.5:
        return (f"[{v1}] ({v2})").center(length, ' ')
    return (f" {v1}  ({v2})").center(length, ' ')
    # }}}

def dci_si_method(image_path, method):
    # {{{
    import shutil
    import tempfile
    import aletheialib.stegosim
    import aletheialib.models
    import numpy as np
    from imageio import imread

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    fn_sim=aletheialib.stegosim.embedding_fn(method)
    method = method.replace("-sim", "")

    embed_fn_saving = False
    if method in aletheialib.stegosim.EMB_FN_SAVING_METHODS:
        embed_fn_saving = True

    # Make some replacements to adapt the name of the method with the name
    # of the model file
    method = method.replace("-color", "")
    method = method.replace("j-uniward", "juniw")
    method = method.replace("s-uniward", "uniw")


    A_dir=tempfile.mkdtemp()
    shutil.copy(image_path, A_dir)

    B_dir=tempfile.mkdtemp()
    aletheialib.stegosim.embed_message(fn_sim, A_dir, "0.40", B_dir, 
                                       embed_fn_saving=embed_fn_saving, show_debug_info=False)

    A_nn = aletheialib.models.NN("effnetb0")
    B_nn = aletheialib.models.NN("effnetb0")
    A_files = glob.glob(os.path.join(A_dir, '*'))
    B_files = glob.glob(os.path.join(B_dir, '*'))

    if len(B_files)==0:
        print("ERROR: Cannot prepare the B set", image_path, B_files, method, "embed_fn_saving=", embed_fn_saving)
        sys.exit(0)

    A_nn = load_model(A_nn, "effnetb0-A-alaska2-"+method)
    B_nn = load_model(B_nn, "effnetb0-B-alaska2-"+method)

    A = imread(A_files[0])
    B = imread(B_files[0])
    # This function must support images with variable size
    # Note that with big images we are only analyzing a small part
    d0 = min(A.shape[0], A_nn.shape[0])
    d1 = min(A.shape[1], A_nn.shape[1])
    d2 = min(A.shape[2], A_nn.shape[2])
    Ai = np.zeros(A_nn.shape)
    Bi = np.zeros(A_nn.shape)
    Ai[:d0, :d1, :d2] = A[:d0, :d1, :d2]
    Bi[:d0, :d1, :d2] = B[:d0, :d1, :d2]

    aa = []
    ab = []
    bb = []
    ba = []
    for flip in [0, 1]:
        for rot in [0, 1, 2, 3]:
            for roll in [0, 128, 256, 384]:

                if flip:
                    Ai = np.flip(Ai)
                    Bi = np.flip(Bi)
                if rot:
                    Ai = np.rot90(Ai, rot)
                    Bi = np.rot90(Bi, rot)
                if roll:
                    Ai = np.roll(Ai, roll, axis=1)
                    Bi = np.roll(Bi, roll, axis=1)

                Ax = np.array([Ai]).astype('float32')/255
                Bx = np.array([Bi]).astype('float32')/255

                p_aa = A_nn.model.predict(Ax, verbose=0)[0,1]
                p_ab = A_nn.model.predict(Bx, verbose=0)[0,1]
                p_bb = B_nn.model.predict(Bx, verbose=0)[0,1]
                p_ba = B_nn.model.predict(Ax, verbose=0)[0,1]
                aa.append(p_aa)
                ab.append(p_ab)
                bb.append(p_bb)
                ba.append(p_ba)

    aa_mean = np.mean(aa)
    aa = np.array(np.round(aa)).astype('uint8')
    ab = np.array(np.round(ab)).astype('uint8')
    bb = np.array(np.round(bb)).astype('uint8')
    ba = np.array(np.round(ba)).astype('uint8')

    inc = ( (aa!=bb) | (ba!=0) | (ab!=1) ).astype('uint8') 
    dci_pred_score = round(1-float(np.sum(inc==1))/(2*len(aa)),3)


    shutil.rmtree(A_dir)
    shutil.rmtree(B_dir)

    return dci_pred_score, aa_mean
    # }}}

def _dci_si_method(image_path, method): # fake
    return np.random.uniform(0.5, 1), np.random.uniform(0.0, 1)

def load_model(nn, model_name):
    # {{{
    # Get the directory where the models are installed
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, os.pardir, os.pardir, 'aletheia-models')

    model_path = os.path.join(dir_path, model_name+".h5")
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}\n")
        sys.exit(-1)
    nn.load_model(model_path, quiet=True)
    # }}}
    return nn

# {{{ _auto()
def _auto():

    if len(sys.argv)<3:
        print(sys.argv[0], "auto <image|dir> [dev]\n")
        sys.exit(0)

    if len(sys.argv)<4:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[3]

    if dev_id == "CPU":
        print("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    import aletheialib.models

    path = aletheialib.utils.absolute_path(sys.argv[2])

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.*'))
    else:
        files = [path]

    nn = aletheialib.models.NN("effnetb0")
    files = nn.filter_images(files)
    if len(files)==0:
        print("ERROR: please provide valid files")
        sys.exit(0)


    jpg_files = []
    bitmap_files = []
    for f in files:
        _, ext = os.path.splitext(f)
        if ext.lower() in ['.jpg', '.jpeg']:
            jpg_files.append(f)
        else:
            bitmap_files.append(f)

    # TODO: Find model paths

    # JPG
    if len(jpg_files)>0:

        nn = load_model(nn, "effnetb0-A-alaska2-outguess")
        outguess_pred = nn.predict(jpg_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-steghide")
        steghide_pred = nn.predict(jpg_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-nsf5")
        nsf5_pred = nn.predict(jpg_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-juniw")
        juniw_pred = nn.predict(jpg_files, 10)


        mx = 20
        print("")
        print(' '*mx + " Outguess  Steghide   nsF5  J-UNIWARD *")
        print('-'*mx + "---------------------------------------")
        for i in range(len(jpg_files)):
            name = os.path.basename(jpg_files[i])
            if len(name)>mx:
                name = name[:mx-3]+"..."
            else:
                name = name.ljust(mx, ' ')
            
            print(name, 
                  _format_line(outguess_pred[i], 9),
                  _format_line(steghide_pred[i], 8),
                  _format_line(nsf5_pred[i], 8),
                  _format_line(juniw_pred[i], 8),
                  )



    # BITMAP
    if len(bitmap_files)>0:

        nn = load_model(nn, "effnetb0-A-alaska2-lsbr")
        lsbr_pred = nn.predict(bitmap_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-lsbm")
        lsbm_pred = nn.predict(bitmap_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-steganogan")
        steganogan_pred = nn.predict(bitmap_files, 10)

        nn = load_model(nn, "effnetb0-A-alaska2-hill")
        hill_pred = nn.predict(bitmap_files, 10)



        mx = 20
        print("")
        print(' '*mx + "   LSBR      LSBM  SteganoGAN  HILL *")
        print('-'*mx + "-------------------------------------")
        for i in range(len(bitmap_files)):
            name = os.path.basename(bitmap_files[i])
            if len(name)>mx:
                name = name[:mx-3]+"..."
            else:
                name = name.ljust(mx, ' ')
            
            print(name, 
                  _format_line(lsbr_pred[i], 10),
                  _format_line(lsbm_pred[i], 8),
                  _format_line(steganogan_pred[i], 8),
                  _format_line(hill_pred[i], 8),
                  )


    print("")
    print("* Probability of steganographic content using the indicated method.\n")
# }}}

# {{{ auto() (v2: dci-si)
def auto():

    if len(sys.argv)<3:
        print(sys.argv[0], "auto <image|dir> [dev]\n")
        sys.exit(0)

    if len(sys.argv)<4:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[3]

    if dev_id == "CPU":
        print("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    import aletheialib.models
    import aletheialib.octave_interface as O

    path = aletheialib.utils.absolute_path(sys.argv[2])

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.*'))
    else:
        files = [path]

    nn = aletheialib.models.NN("effnetb0")
    files = nn.filter_images(files)
    if len(files)==0:
        print("ERROR: please provide valid files")
        sys.exit(0)


    jpg_files = []
    bitmap_files = []
    for f in files:
        _, ext = os.path.splitext(f)
        if ext.lower() in ['.jpg', '.jpeg']:
            jpg_files.append(f)
        else:
            bitmap_files.append(f)


    # JPG
    if len(jpg_files)>0:

        download_octave_jpeg_toolbox()
        download_octave_code("NSF5_COLOR")
        download_octave_code("J_UNIWARD_COLOR")
        aletheialib.utils.check_bin("outguess")
        aletheialib.utils.check_bin("steghide")

        mx = 20
        print("")
        print(' '*mx + "   Outguess      Steghide      nsF5          J-UNIWARD * ")
        print('-'*mx + "---------------------------------------------------------")
        for i in range(len(jpg_files)):
            name = os.path.basename(jpg_files[i])
            if len(name)>mx:
                name = name[:mx-3]+"..."
            else:
                name = name.ljust(mx, ' ')
 
            outguess_dci, outguess_pred = dci_si_method(jpg_files[i], "outguess-sim")
            steghide_dci, steghide_pred = dci_si_method(jpg_files[i], "steghide-sim")
            nsf5_dci, nsf5_pred = dci_si_method(jpg_files[i], "nsf5-color-sim")
            juniw_dci, juniw_pred = dci_si_method(jpg_files[i], "j-uniward-color-sim")

            print(name, 
                  _format_line_2(outguess_pred, outguess_dci, 13),
                  _format_line_2(steghide_pred, steghide_dci, 13),
                  _format_line_2(nsf5_pred, nsf5_dci, 13),
                  _format_line_2(juniw_pred, juniw_dci, 13),
                  )



    # BITMAP
    if len(bitmap_files)>0:

        download_octave_code("S_UNIWARD_COLOR")
        download_octave_code("HILL_COLOR")
        aletheialib.utils.check_bin("steganogan")

        mx = 20
        print("")
        print(' '*mx + "  LSBR^   LSBM          SteganoGAN    HILL          UNIWARD *")
        print('-'*mx + "-------------------------------------------------------------")
        for i in range(len(bitmap_files)):
            name = os.path.basename(bitmap_files[i])
            if len(name)>mx:
                name = name[:mx-3]+"..."
            else:
                name = name.ljust(mx, ' ')
 
            beta = O._attack('WS', bitmap_files[i], params={"channel":1})["data"][0][0]
            if beta<=0.05:
                beta = 0
     
            lsbr_pred = beta*2
            lsbm_dci, lsbm_pred = dci_si_method(bitmap_files[i], "lsbm-sim")
            steganogan_dci, steganogan_pred = dci_si_method(bitmap_files[i], "steganogan-sim")
            hill_dci, hill_pred = dci_si_method(bitmap_files[i], "hill-color-sim")
            uniw_dci, uniw_pred = dci_si_method(bitmap_files[i], "s-uniward-color-sim")

            print(name, 
                  _format_line(lsbr_pred, 6, 0),
                  _format_line_2(lsbm_pred, lsbm_dci, 13),
                  _format_line_2(steganogan_pred, steganogan_dci, 13),
                  _format_line_2(hill_pred, hill_dci, 13),
                  _format_line_2(uniw_pred, uniw_dci, 13),
                  )


    print("")
    print(" *  Probability of steganographic content using the indicated method.")
    print("( ) Accuracy (confidence) of the results using the DCI-SI technique.")   
    print(" ^  In LSBR the estimated payload is shown instead of the probability.")
    print("")

# }}}

# {{{ dci()
def dci():

    if len(sys.argv)<4:
        print(sys.argv[0], "dci <sim> <img dir> [dev]\n")
        print("Example:");
        print(sys.argv[0], "dci steghide-sim images/\n")
        sys.exit(0)

    files = glob.glob(os.path.join(sys.argv[3], '*'))

    if len(sys.argv)<5:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[4]

    if dev_id == "CPU":
        print("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if not os.path.isdir(sys.argv[3]):
        print("ERROR: Please, provide a valid directory\n")
        sys.exit(0)

    if len(files)<10:
        print("ERROR: We need more images from the same actor\n")
        sys.exit(0)

    
    ext = os.path.splitext(files[0])[1].lower().replace('.jpeg', '.jpg')
    for f in files:
        curr_ext = os.path.splitext(f)[1].lower().replace('.jpeg', '.jpg')
        if ext != curr_ext:
            print(f"ERROR: All images must be of the same type: {curr_ext}!={ext} \n")
            sys.exit(0)

    embed_fn_saving = False
    if ext=='.jpg':
        embed_fn_saving = True


    import shutil
    import tempfile
    import aletheialib.stegosim
    import aletheialib.models
    import numpy as np

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    A_nn = aletheialib.models.NN("effnetb0")
    A_files = files

    fn_sim=aletheialib.stegosim.embedding_fn(sys.argv[2])
    method = sys.argv[2]
    method = method.replace("-sim", "")

    B_dir=tempfile.mkdtemp()
    print("Preparind the B set ...")
    aletheialib.stegosim.embed_message(fn_sim, sys.argv[3], "0.40", B_dir, 
                                       embed_fn_saving=embed_fn_saving)

    B_nn = aletheialib.models.NN("effnetb0")
    B_files = glob.glob(os.path.join(B_dir, '*'))

    # Make some replacements to adapt the name of the method with the name
    # of the model file
    method = method.replace("-color", "")
    method = method.replace("j-uniward", "juniw")

    A_nn = load_model(A_nn, "effnetb0-A-alaska2-"+method)
    B_nn = load_model(B_nn, "effnetb0-B-alaska2-"+method)

    # Predictions for the DCI method
    p_aa = A_nn.predict(A_files, 10)
    p_ab = A_nn.predict(B_files, 10)
    p_bb = B_nn.predict(B_files, 10)
    p_ba = B_nn.predict(A_files, 10)

    p_aa = np.round(p_aa).astype('uint8')
    p_ab = np.round(p_ab).astype('uint8')
    p_ba = np.round(p_ba).astype('uint8')
    p_bb = np.round(p_bb).astype('uint8')

    # Inconsistencies
    inc = ( (p_aa!=p_bb) | (p_ba!=0) | (p_ab!=1) ).astype('uint8')
    inc1 = (p_aa!=p_bb).astype('uint8')
    inc2 = ( (p_ba!=0) | (p_ab!=1) ).astype('uint8')
    inc2c = (p_ab!=1).astype('uint8')
    inc2s = (p_ba!=0).astype('uint8')


    for i in range(len(p_aa)):
        r = ""
        if inc[i]:
            r = str(round(p_aa[i],3))+" (inc)"
        else:
            r = round(p_aa[i],3)
        print(A_files[i], "\t", r)

    """
    print("#inc:", np.sum(inc==1), "#incF1:", np.sum(inc1==1), "#incF2:", np.sum(inc2==1),
           "#incF2C", np.sum(inc2c), "#incF2S:", np.sum(inc2s))
    print("#no_inc:", len(A_files)-np.sum(inc==1))
    print("--")
    """
    print("DCI prediction score:", round(1-float(np.sum(inc==1))/(2*len(p_aa)),3))

    shutil.rmtree(B_dir)
# }}}

# {{{ dci_si()
def dci_si():

    if len(sys.argv)<4:
        print(sys.argv[0], "dci-si <sim> <image|dir> [dev]\n")
        print("Example:");
        print(sys.argv[0], "dci-si steghide-sim image.jpg\n")
        sys.exit(0)

    method = sys.argv[2]
    path = sys.argv[3]

    if len(sys.argv)<5:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[4]

    if dev_id == "CPU":
        print("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    from aletheialib.stegosim import JPEG_METHODS 
    import aletheialib.models # XXX Print logs here

    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*.*'))
    else:
        files = [path]

    mx = 20

    print("")
    print('     IMAGE                  DCI-SI     PREDICTION ')
    print('  --------------------------------------------------')
    for image_path in files:

        # Fix uniward name
        method = method.replace("juniw", "j-uniward")

        # Check pair method/image
        ext = os.path.splitext(image_path)[1].lower().replace('.jpeg', '.jpg')
        m = method.replace('-sim', '').replace('-', '_')
        if ext=='.jpg' and m not in JPEG_METHODS:
            print("ERROR: Please, provide a compatible image for this method\n")
            sys.exit(0)
        elif ext!='.jpg' and m  in JPEG_METHODS:
            print("ERROR: Please, provide a compatible image for this method\n")
            sys.exit(0)

        dci_pred_score, aa_mean = dci_si_method(image_path, method)

        textpred = "cover"
        if aa_mean > 0.5:
            textpred = "stego"

        name = os.path.basename(image_path)
        if len(name)>mx:
            name = name[:mx-3]+"..."
        else:
            name = name.ljust(mx, ' ')

        print(f"     {name}   {dci_pred_score:.3f}      {aa_mean:.3f} ({textpred})")

    print("")
    print("  DCI-SI: Accuracy (confidence) of the model.")   
    print("  PREDICTION: Probability of steganographic content using the indicated method.")   
    print("")
# }}}







