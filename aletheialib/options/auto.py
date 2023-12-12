
import os
import sys
import glob
import aletheialib.utils


doc = "\n" \
"  Automated tools:\n" \
"  - auto:      Try different steganalysis methods.\n" \
"  - dci:       Predicts a set of images using DCI evaluation."



def _format_line(value, length):
    if value > 0.5:
        return ("["+str(round(value,1))+"]").center(length, ' ')

    return str(round(value,1)).center(length, ' ')

def load_model(nn, model_name):

    # Get the directory where the models are installed
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, os.pardir, os.pardir, 'aletheia-models')

    model_path = os.path.join(dir_path, model_name+".h5")
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}\n")
        sys.exit(-1)
    nn.load_model(model_path, quiet=True)
    return nn

# {{{ auto()
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






