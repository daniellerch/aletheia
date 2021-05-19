
import os
import sys
import glob
import aletheialib.utils


doc = "\n" \
"  Automatic steganalysis:\n" \
"  - auto:      Try different steganalysis methods."
#"  - auto-dci:      Try different steganalysis methods with DCI."



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

def launch():

    if len(sys.argv)!=3:
        print(sys.argv[0], "auto <image|dir>\n")
        sys.exit(0)

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
        print("ERROR: please provice valid files")
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
    print("* Probability of being stego using the indicated steganographic method.\n")



