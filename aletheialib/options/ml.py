import os
import sys
import glob
import shutil
import random
import numpy as np
from aletheialib.utils import download_e4s

doc="\n" \
"  ML-based steganalysis:\n" \
"  - split-sets:            Prepare sets for training and testing.\n" \
"  - split-sets-dci:        Prepare sets for training and testing (DCI).\n" \
"  - create-actors:         Prepare actors for training and testing.\n" \
"  - effnetb0:              Train a model with EfficientNet B0.\n" \
"  - effnetb0-score:        Score with EfficientNet B0.\n" \
"  - effnetb0-predict:      Predict with EfficientNet B0.\n" \
"  - effnetb0-dci-score:    DCI Score with EfficientNet B0.\n" \
"  - effnetb0-dci-predict:  DCI Prediction with EfficientNet B0.\n" \
"  - e4s:                   Train Ensemble Classifiers for Steganalysis.\n" \
"  - e4s-predict:           Predict using EC.\n" \
"  - actor-predict-fea:     Predict features for an actor.\n" \
"  - actors-predict-fea:    Predict features for a set of actors."



# {{{ split_sets
def split_sets():

    if len(sys.argv)<8:
        print(sys.argv[0], "split-sets <cover-dir> <stego-dir> <output-dir> <#valid> <#test>\n")
        print("     cover-dir:    Directory containing cover images")
        print("     stego-dir:    Directory containing stego images")
        print("     output-dir:   Output directory. Three sets will be created")
        print("     #valid:       Number of images for the validation set")
        print("     #test:        Number of images for the testing set")
        print("     seed:         Seed for reproducible results")
        print("")
        sys.exit(0)

    cover_dir=sys.argv[2]
    stego_dir=sys.argv[3]
    output_dir=sys.argv[4]
    n_valid=int(sys.argv[5])
    n_test=int(sys.argv[6])
    seed=int(sys.argv[7])


    cover_files = np.array(sorted(glob.glob(os.path.join(cover_dir, '*'))))
    stego_files = np.array(sorted(glob.glob(os.path.join(stego_dir, '*'))))

    if len(cover_files)!=len(stego_files):
        print("ERROR: we expect the same number of cover and stego files");
        sys.exit(0)


    indices = list(range(len(cover_files)))
    random.seed(seed)
    random.shuffle(indices)

    valid_indices = indices[:n_valid//2]
    test_C_indices = indices[n_valid//2:n_valid//2+n_test//2]
    test_S_indices = indices[n_valid//2+n_test//2:n_valid//2+n_test]
    train_indices = indices[n_valid//2+n_test:]

    train_C_dir = os.path.join(output_dir, "train", "cover")
    train_S_dir = os.path.join(output_dir, "train", "stego")
    valid_C_dir = os.path.join(output_dir, "valid", "cover")
    valid_S_dir = os.path.join(output_dir, "valid", "stego")
    test_C_dir = os.path.join(output_dir, "test", "cover")
    test_S_dir = os.path.join(output_dir, "test", "stego")


    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    os.makedirs(train_C_dir, exist_ok=True)
    os.makedirs(train_S_dir, exist_ok=True)
    os.makedirs(valid_C_dir, exist_ok=True)
    os.makedirs(valid_S_dir, exist_ok=True)
    os.makedirs(test_C_dir, exist_ok=True)
    os.makedirs(test_S_dir, exist_ok=True)
    
    for f in cover_files[train_indices]:
        shutil.copy(f, train_C_dir)

    for f in stego_files[train_indices]:
        shutil.copy(f, train_S_dir)

    for f in cover_files[valid_indices]:
        shutil.copy(f, valid_C_dir)

    for f in stego_files[valid_indices]:
        shutil.copy(f, valid_S_dir)

    for f in cover_files[test_C_indices]:
        shutil.copy(f, test_C_dir)

    for f in stego_files[test_S_indices]:
        shutil.copy(f, test_S_dir)
# }}}

# {{{ split_sets_dci
def split_sets_dci():

    if len(sys.argv)<9:
        print(sys.argv[0], "split-sets <cover-dir> <stego-dir> <double-dir> <output-dir> <#valid> <#test> <seed>\n")
        print("     cover-dir:    Directory containing cover images")
        print("     stego-dir:    Directory containing stego images")
        print("     double-dir:   Directory containing double stego images")
        print("     output-dir:   Output directory. Three sets will be created")
        print("     #valid:       Number of images for the validation set")
        print("     #test:        Number of images for the testing set")
        print("     seed:         Seed for reproducible results")
        print("")
        sys.exit(0)

    cover_dir=sys.argv[2]
    stego_dir=sys.argv[3]
    double_dir=sys.argv[4]
    output_dir=sys.argv[5]
    n_valid=int(sys.argv[6])
    n_test=int(sys.argv[7])
    seed=int(sys.argv[8])


    cover_files = np.array(sorted(glob.glob(os.path.join(cover_dir, '*'))))
    stego_files = np.array(sorted(glob.glob(os.path.join(stego_dir, '*'))))
    double_files = np.array(sorted(glob.glob(os.path.join(double_dir, '*'))))

    if len(cover_files)!=len(stego_files) or len(stego_files)!=len(double_files):
        print("split-sets-dci error: we expect sets with the same number of images")
        sys.exit(0)

    indices = list(range(len(cover_files)))
    random.seed(seed)
    random.shuffle(indices)

    valid_indices = indices[:n_valid//2]
    test_C_indices = indices[n_valid//2:n_valid//2+n_test//2]
    test_S_indices = indices[n_valid//2+n_test//2:n_valid//2+n_test]
    train_indices = indices[n_valid//2+n_test:]


    A_train_C_dir = os.path.join(output_dir, "A_train", "cover")
    A_train_S_dir = os.path.join(output_dir, "A_train", "stego")
    A_valid_C_dir = os.path.join(output_dir, "A_valid", "cover")
    A_valid_S_dir = os.path.join(output_dir, "A_valid", "stego")
    A_test_C_dir = os.path.join(output_dir, "A_test", "cover")
    A_test_S_dir = os.path.join(output_dir, "A_test", "stego")
    B_train_S_dir = os.path.join(output_dir, "B_train", "stego")
    B_train_D_dir = os.path.join(output_dir, "B_train", "double")
    B_valid_S_dir = os.path.join(output_dir, "B_valid", "stego")
    B_valid_D_dir = os.path.join(output_dir, "B_valid", "double")
    B_test_S_dir = os.path.join(output_dir, "B_test", "stego")
    B_test_D_dir = os.path.join(output_dir, "B_test", "double")


    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    os.makedirs(A_train_C_dir, exist_ok=True)
    os.makedirs(A_train_S_dir, exist_ok=True)
    os.makedirs(A_valid_C_dir, exist_ok=True)
    os.makedirs(A_valid_S_dir, exist_ok=True)
    os.makedirs(A_test_C_dir, exist_ok=True)
    os.makedirs(A_test_S_dir, exist_ok=True)
    os.makedirs(B_train_S_dir, exist_ok=True)
    os.makedirs(B_train_D_dir, exist_ok=True)
    os.makedirs(B_valid_S_dir, exist_ok=True)
    os.makedirs(B_valid_D_dir, exist_ok=True)
    os.makedirs(B_test_S_dir, exist_ok=True)
    os.makedirs(B_test_D_dir, exist_ok=True)

    for f in cover_files[train_indices]:
        shutil.copy(f, A_train_C_dir)

    for f in stego_files[train_indices]:
        shutil.copy(f, A_train_S_dir)
        shutil.copy(f, B_train_S_dir)

    for f in double_files[train_indices]:
        shutil.copy(f, B_train_D_dir)


    for f in cover_files[valid_indices]:
        shutil.copy(f, A_valid_C_dir)

    for f in stego_files[valid_indices]:
        shutil.copy(f, A_valid_S_dir)
        shutil.copy(f, B_valid_S_dir)

    for f in double_files[valid_indices]:
        shutil.copy(f, B_valid_D_dir)


    for f in cover_files[test_C_indices]:
        shutil.copy(f, A_test_C_dir)

    for f in stego_files[test_S_indices]:
        shutil.copy(f, A_test_S_dir)

    for f in stego_files[test_C_indices]:
        shutil.copy(f, B_test_S_dir)

    for f in double_files[test_S_indices]:
        shutil.copy(f, B_test_D_dir)



# }}}

# {{{ create_actors
def create_actors():

    if len(sys.argv)<8:
        print(sys.argv[0], "create-actors <cover-dir> <stego-dir> <output-dir> <#innocent> <#guilty> <min-max> <seed>\n")
        print("     cover-dir:    Directory containing cover images")
        print("     stego-dir:    Directory containing stego images")
        print("     output-dir:   Output directory where actors will be created")
        print("     #innocent     Number of innocent actors")
        print("     #guilty:      Number of guilty actors")
        print("     min-max:      Range of images for each actor")
        print("     seed:         Seed for reproducible results")
        print("")
        sys.exit(0)

    cover_dir=sys.argv[2]
    stego_dir=sys.argv[3]
    output_dir=sys.argv[4]
    n_innocent=int(sys.argv[5])
    n_guilty=int(sys.argv[6])
    min_max = sys.argv[7].split('-')
    if len(min_max)!=2:
        print("create-actors error: min-max format error, use for example: 10-50")
        sys.exit(0)
    seed=int(sys.argv[8])

    cover_files = sorted(glob.glob(os.path.join(cover_dir, '*')))
    stego_files = sorted(glob.glob(os.path.join(stego_dir, '*')))

    if len(cover_files)!=len(stego_files):
        print("ERROR: we expect the same number of cover and stego files");
        sys.exit(0)


    innocent_actors = os.path.join(output_dir, "innocent")
    guilty_actors = os.path.join(output_dir, "guilty")

    os.makedirs(innocent_actors, exist_ok=True)
    os.makedirs(guilty_actors, exist_ok=True)
 

    random.seed(seed)

    # Innocent actors
    for i in range(1, n_innocent+1):
        num_actor_images = random.randint(int(min_max[0]), int(min_max[1]))
        actor_dir = os.path.join(innocent_actors, f"actor{i}")

        if os.path.isdir(actor_dir):
           print(f"Innocent actor already exists actor{i}")
           continue

        if not os.path.isdir(actor_dir):
            os.mkdir(actor_dir)

        for j in range(1, num_actor_images+1):
            # random image
            k = random.randint(0, len(cover_files)-1)
            src_file = cover_files[k]
            n, ext = os.path.splitext(src_file)
            dst_file = os.path.join(innocent_actors, f"actor{i}/{j}{ext}")
            if os.path.exists(dst_file):
                print(f"Already exists actor{i}/{j}{ext}")
            else:
                shutil.copy(src_file, dst_file)
                print(f"Copy {src_file} actor{i}/{j}{ext}")


    # Guilty actors
    for i in range(1, n_guilty+1):
        num_actor_images = random.randint(int(min_max[0]), int(min_max[1]))
        actor_dir = os.path.join(guilty_actors, f"actor{i}")

        if os.path.isdir(actor_dir):
           print(f"Guilty actor already exists actor{i}")
           continue

        f = open(os.path.join(guilty_actors, f"actor{i}.txt"), 'w')

        if not os.path.isdir(actor_dir):
            os.mkdir(actor_dir)

        for j in range(1, num_actor_images+1):

            # random image
            k = random.randint(0, len(cover_files)-1)

            prob_stego = round(random.uniform(0.1, 1), 2)
            if random.random()<prob_stego:
                src_file = stego_files[k]
                n, ext = os.path.splitext(src_file)
                f.write(f'{j}{ext}, 1\n')
            else:
                src_file = cover_files[k]
                n, ext = os.path.splitext(src_file)
                f.write(f'{j}{ext}, 0\n')

            dst_file = os.path.join(guilty_actors, f"actor{i}/{j}{ext}")

            if os.path.exists(dst_file):
                print(f"Already exists actor{i}/{j}{ext}")
            else:
                shutil.copy(src_file, dst_file)
                print(f"Copy {src_file} actor{i}/{j}{ext}")

        f.close()


# }}}


# {{{ effnetb0
def effnetb0():

    if len(sys.argv)<7:
        print(sys.argv[0], "effnetb0 <trn-cover-dir> <trn-stego-dir> <val-cover-dir> <val-stego-dir> <model-file> [dev] [ES] [BS]\n")
        print("     trn-cover-dir:    Directory containing training cover images")
        print("     trn-stego-dir:    Directory containing training stego images")
        print("     val-cover-dir:    Directory containing validation cover images")
        print("     val-stego-dir:    Directory containing validation stego images")
        print("     model-name:       A name for the model")
        print("     dev:        Device: GPU Id or 'CPU' (default='CPU')")
        print("     ES:         early stopping iterations x1000 (default=10)")
        print("     BS:         Batch size (default=16)")
        print("")
        sys.exit(0)

    trn_cover_dir=sys.argv[2]
    trn_stego_dir=sys.argv[3]
    val_cover_dir=sys.argv[4]
    val_stego_dir=sys.argv[5]
    model_name=sys.argv[6]

    if len(sys.argv)<8:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[7]

    if len(sys.argv)<9:
        early_stopping = 10
        print("'ES' not provided, using:", early_stopping)
    else:
        early_stopping = int(sys.argv[8])

    if len(sys.argv)<10:
        batch = 16
        print("'BS' not provided, using:", batch)
    else:
        batch = int(sys.argv[9])



    if dev_id == "CPU":
        print("Running with CPU. It could be very slow!")


    os.environ["CUDA_VISIBLE_DEVICES"]=dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


    trn_cover_files = sorted(glob.glob(os.path.join(trn_cover_dir, '*')))
    trn_stego_files = sorted(glob.glob(os.path.join(trn_stego_dir, '*')))
    val_cover_files = sorted(glob.glob(os.path.join(val_cover_dir, '*')))
    val_stego_files = sorted(glob.glob(os.path.join(val_stego_dir, '*')))

    print("train:", len(trn_cover_files),"+",len(trn_stego_files))
    print("valid:", len(val_cover_files),"+",len(val_stego_files))

    if (not len(trn_cover_files) or not len(trn_stego_files) or
        not len(val_cover_files) or not len(val_stego_files)):
        print("ERROR: directory without files found")
        sys.exit(0)


    import aletheialib.models
    nn = aletheialib.models.NN("effnetb0", model_name=model_name, shape=(512,512,3))
    nn.train(trn_cover_files, trn_stego_files, batch, # 36|40
    #nn = aletheialib.models.NN("effnetb0", model_name=model_name, shape=(32,32,3))
    #nn.train(trn_cover_files, trn_stego_files, 500, # 36|40
             val_cover_files, val_stego_files, 10,
             1000000, early_stopping)


# }}}

# {{{ effnetb0_score
def effnetb0_score():

    if len(sys.argv)<5:
        print(sys.argv[0], "effnetb0-score <test-cover-dir> <test-stego-dir> <model-file> [dev]\n")
        print("     test-cover-dir:    Directory containing cover images")
        print("     test-stego-dir:    Directory containing stego images")
        print("     model-file:        Path of the model")
        print("     dev:        Device: GPU Id or 'CPU' (default='CPU')")
        print("")
        sys.exit(0)

    import aletheialib.models

    cover_dir=sys.argv[2]
    stego_dir=sys.argv[3]
    model_file=sys.argv[4]

    if len(sys.argv)<6:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[5]

    if dev_id == "CPU":
        print("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    cover_files = sorted(glob.glob(os.path.join(cover_dir, '*')))
    stego_files = sorted(glob.glob(os.path.join(stego_dir, '*')))

    nn = aletheialib.models.NN("effnetb0")
    nn.load_model(model_file)

    pred_cover = nn.predict(cover_files, 10)
    pred_stego = nn.predict(stego_files, 10)

    ok = np.sum(np.round(pred_cover)==0)+np.sum(np.round(pred_stego)==1)
    score = ok/(len(pred_cover)+len(pred_stego))


    print("score:", score)

# }}}

# {{{ effnetb0_predict
def effnetb0_predict():

    if len(sys.argv)<4:
        print(sys.argv[0], "effnetb0-predict <test-dir/image> <model-file> [dev]\n")
        print("     test-dir:    Directory containing test images")
        print("     model-file:        Path of the model")
        print("     dev:        Device: GPU Id or 'CPU' (default='CPU')")
        print("")
        sys.exit(0)

    import aletheialib.models

    test_dir=sys.argv[2]
    model_file=sys.argv[3]

    if len(sys.argv)<5:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[4]

    if dev_id == "CPU":
        print("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    nn = aletheialib.models.NN("effnetb0")
    nn.load_model(model_file)

    if os.path.isdir(test_dir):
        test_files = sorted(glob.glob(os.path.join(test_dir, '*')))
    else:
        test_files = [test_dir]

    test_files = nn.filter_images(test_files)
    if len(test_files)==0:
        print("ERROR: please provice valid files")
        sys.exit(0)


    pred = nn.predict(test_files, 10)

    for i in range(len(pred)):
        print(test_files[i], round(pred[i],3))

# }}}

# {{{ effnetb0_dci_score
def effnetb0_dci_score():

    if len(sys.argv)<8:
        print(sys.argv[0], "effnetb0-dci-score <A-test-cover-dir> <A-test-stego-dir> <B-test-stego-dir> <B-test-double-dir> <A-model-file> <B-model-file> [dev]\n")
        print("     A-test-cover-dir:    Directory containing A-cover images")
        print("     A-test-stego-dir:    Directory containing A-stego images")
        print("     B-test-stego-dir:    Directory containing B-stego images")
        print("     B-test-double-dir:   Directory containing B-double images")
        print("     A-model-file:        Path of the A-model")
        print("     B-model-file:        Path of the B-model")
        print("     dev:                 Device: GPU Id or 'CPU' (default='CPU')")
        print("")
        sys.exit(0)

    import aletheialib.models
    from sklearn.metrics import accuracy_score

    A_cover_dir=sys.argv[2]
    A_stego_dir=sys.argv[3]
    B_stego_dir=sys.argv[4]
    B_double_dir=sys.argv[5]
    A_model_file=sys.argv[6]
    B_model_file=sys.argv[7]

    if len(sys.argv)<9:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[8]

    if dev_id == "CPU":
        print("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    A_cover_files = sorted(glob.glob(os.path.join(A_cover_dir, '*')))
    A_stego_files = sorted(glob.glob(os.path.join(A_stego_dir, '*')))
    B_stego_files = sorted(glob.glob(os.path.join(B_stego_dir, '*')))
    B_double_files = sorted(glob.glob(os.path.join(B_double_dir, '*')))

    A_nn = aletheialib.models.NN("effnetb0")
    A_nn.load_model(A_model_file)
    B_nn = aletheialib.models.NN("effnetb0")
    B_nn.load_model(B_model_file)


    A_files = A_cover_files+A_stego_files
    B_files = B_stego_files+B_double_files


    p_aa_ = A_nn.predict(A_files, 10)
    p_ab_ = A_nn.predict(B_files, 10)
    p_bb_ = B_nn.predict(B_files, 10)
    p_ba_ = B_nn.predict(A_files, 10)

    p_aa = np.round(p_aa_).astype('uint8')
    p_ab = np.round(p_ab_).astype('uint8')
    p_ba = np.round(p_ba_).astype('uint8')
    p_bb = np.round(p_bb_).astype('uint8')

    y_true = np.array([0]*len(A_cover_files) + [1]*len(A_stego_files))
    inc = ( (p_aa!=p_bb) | (p_ba!=0) | (p_ab!=1) ).astype('uint8')
    inc1 = (p_aa!=p_bb).astype('uint8')
    inc2 = ( (p_ba!=0) | (p_ab!=1) ).astype('uint8')
    inc2c = (p_ab!=1).astype('uint8')
    inc2s = (p_ba!=0).astype('uint8')
    C_ok = ( (p_aa==0) & (p_aa==y_true) & (inc==0) ).astype('uint8')
    S_ok = ( (p_aa==1) & (p_aa==y_true) & (inc==0) ).astype('uint8')

    tp = np.sum( (p_aa==1) & (p_aa==y_true) ) 
    tn = np.sum( (p_aa==0) & (p_aa==y_true) ) 
    fp = np.sum( (p_aa==1) & (p_aa!=y_true) ) 
    fn = np.sum( (p_aa==0) & (p_aa!=y_true) ) 
    print(f"aa-confusion-matrix tp: tp={tp}, tn={tn}, fp={fp}, fn={fn}")

    tp = np.sum( (p_aa==1) & (p_aa==y_true) & (inc==0) ) 
    tn = np.sum( (p_aa==0) & (p_aa==y_true) & (inc==0) ) 
    fp = np.sum( (p_aa==1) & (p_aa!=y_true) & (inc==0) ) 
    fn = np.sum( (p_aa==0) & (p_aa!=y_true) & (inc==0) ) 
    print(f"dci-confusion-matrix tp: tp={tp}, tn={tn}, fp={fp}, fn={fn}")


    print("#inc:", np.sum(inc==1), "#incF1:", np.sum(inc1==1), "#incF2:", np.sum(inc2==1),
           "#incF2C", np.sum(inc2c), "#incF2S:", np.sum(inc2s))
    print("#no_inc:", len(A_files)-np.sum(inc==1))
    print("#C-ok:", np.sum(C_ok==1))
    print("#S-ok:", np.sum(S_ok==1))
    print("aa-score:", accuracy_score(y_true, p_aa))
    print("bb-score:", accuracy_score(y_true, p_bb))
    print("dci-score:", round(float(np.sum(C_ok==1)+np.sum(S_ok==1))/(len(A_files)-np.sum(inc==1)),2))
    print("--")
    print("dci-prediction-score:", round(1-float(np.sum(inc==1))/(2*len(p_aa)),3))

# }}}

# {{{ effnetb0_dci_predict
def effnetb0_dci_predict():

    if len(sys.argv)<6:
        print(sys.argv[0], "effnetb0-dci-predict <A-test-dir> <B-test-dir> <A-model-file> <B-model-file> [dev]\n")
        print("     A-test-dir:          Directory containing A test images")
        print("     B-test-dir:          Directory containing B test images")
        print("     A-model-file:        Path of the A-model")
        print("     B-model-file:        Path of the B-model")
        print("     dev:                 Device: GPU Id or 'CPU' (default='CPU')")
        print("")
        sys.exit(0)

    import aletheialib.models
    from sklearn.metrics import accuracy_score

    A_dir=sys.argv[2]
    B_dir=sys.argv[3]
    A_model_file=sys.argv[4]
    B_model_file=sys.argv[5]

    if len(sys.argv)<7:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[6]

    if dev_id == "CPU":
        print("Running with CPU. It could be very slow!")

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    A_files = sorted(glob.glob(os.path.join(A_dir, '*')))
    B_files = sorted(glob.glob(os.path.join(B_dir, '*')))

    A_nn = aletheialib.models.NN("effnetb0")
    A_nn.load_model(A_model_file)
    B_nn = aletheialib.models.NN("effnetb0")
    B_nn.load_model(B_model_file)



    p_aa = A_nn.predict(A_files, 10)
    p_ab = A_nn.predict(B_files, 10)
    p_bb = B_nn.predict(B_files, 10)
    p_ba = B_nn.predict(A_files, 10)


    p_aa = np.round(p_aa).astype('uint8')
    p_ab = np.round(p_ab).astype('uint8')
    p_ba = np.round(p_ba).astype('uint8')
    p_bb = np.round(p_bb).astype('uint8')

    inc = ( (p_aa!=p_bb) | (p_ba!=0) | (p_ab!=1) ).astype('uint8')
    inc1 = (p_aa!=p_bb).astype('uint8')
    inc2 = ( (p_ba!=0) | (p_ab!=1) ).astype('uint8')
    inc2c = (p_ab!=1).astype('uint8')
    inc2s = (p_ba!=0).astype('uint8')


    for i in range(len(p_aa)):
        r = ""
        if inc[i]:
            r = "INC"
        else:
            r = round(p_aa[i],3)
        print(A_files[i], r)

    print("#inc:", np.sum(inc==1), "#incF1:", np.sum(inc1==1), "#incF2:", np.sum(inc2==1),
           "#incF2C", np.sum(inc2c), "#incF2S:", np.sum(inc2s))
    print("#no_inc:", len(A_files)-np.sum(inc==1))
    print("--")
    print("dci-prediction-score:", round(1-float(np.sum(inc==1))/(2*len(p_aa)),3))

# }}}




# {{{ e4s
def e4s():

    if len(sys.argv)!=5:
        print(sys.argv[0], "e4s <cover-fea> <stego-fea> <model-file>\n")
        sys.exit(0)

    download_e4s()

    import aletheialib.utils
    import aletheialib.models
    from sklearn.model_selection import train_test_split

    cover_fea=sys.argv[2]
    stego_fea=sys.argv[3]
    model_file = aletheialib.utils.absolute_path(sys.argv[4])

    X_cover = pandas.read_csv(cover_fea, delimiter = " ").values
    X_stego = pandas.read_csv(stego_fea, delimiter = " ").values
    #X_cover=numpy.loadtxt(cover_fea)
    #X_stego=numpy.loadtxt(stego_fea)

    X=numpy.vstack((X_cover, X_stego))
    y=numpy.hstack(([0]*len(X_cover), [1]*len(X_stego)))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10)

    clf = aletheialib.models.Ensemble4Stego()
    clf.fit(X_train, y_train)
    val_score=clf.score(X_val, y_val)

    clf.save(model_file)
    print("Validation score:", val_score)
# }}}

# {{{ e4s_predict
def e4s_predict():

    if len(sys.argv)!=5:
        print(sys.argv[0], "e4s-predict <model-file> <feature-extractor> <image/dir>\n")
        print("")
        sys.exit(0)

    download_e4s()

    import aletheialib.models
    import aletheialib.utils
    import aletheialib.feaext

    model_file = sys.argv[2]
    extractor = sys.argv[3]
    path = aletheialib.utils.absolute_path(sys.argv[4])

    files=[]
    if os.path.isdir(path):
        for dirpath,_,filenames in os.walk(path):
            for f in filenames:
                path=os.path.abspath(os.path.join(dirpath, f))
                if not aletheialib.utils.is_valid_image(path):
                    print("Warning, please provide a valid image: ", f)
                else:
                    files.append(path)
    else:
        files=[path]


    clf = aletheialib.models.Ensemble4Stego()
    clf.load(model_file)
    for f in files:
        # TODO: make it multithread
        X = aletheialib.feaext.extractor_fn(extractor)(f)
        X = X.reshape((1, X.shape[0]))
        p = clf.predict(X)
        #print(p)
        if p[0] == 0:
            print(os.path.basename(f), "Cover")
        else:
            print(os.path.basename(f), "Stego")
# }}}

# {{{ actor*_predict_fea()

A_models = {}
B_models = {}


def _load_model(nn, model_name):

    # Get the directory where the models are installed
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, os.pardir, os.pardir, 'aletheia-models')

    model_path = os.path.join(dir_path, model_name+".h5")
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}\n")
        sys.exit(-1)
    nn.load_model(model_path, quiet=False)
    return nn

def _actor(image_dir, output_file, dev_id="cpu"):
    files = glob.glob(os.path.join(image_dir, '*'))

    if not os.path.isdir(image_dir):
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

    if ext=='.jpg':
        simulators = ["outguess-sim", "steghide-sim", "nsf5-color-sim", "j-uniward-color-sim"]
    else:
        simulators = ["steganogan-sim", "lsbm-sim", "hill-color-sim", "s-uniward-color-sim"]

    import gc
    import shutil
    import tempfile
    import aletheialib.utils
    import aletheialib.stegosim
    import aletheialib.models
    import numpy as np
    from tensorflow.python.framework import ops

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    A_nn = aletheialib.models.NN("effnetb0")
    A_files = files

    output = ""
    for simulator in simulators:
        if simulator in ["steganogan-sim", "outguess-sim", "steghide-sim", "nsf5-color-sim", "j-uniward-color-sim"]:
            embed_fn_saving = True
        else:
            embed_fn_saving = False
            
        fn_sim=aletheialib.stegosim.embedding_fn(simulator)

        # Make some replacements to adapt the name of the method with the name
        # of the model file
        method = simulator
        method = method.replace("-sim", "")
        method = method.replace("-color", "")
        method = method.replace("j-uniward", "juniw")
        method = method.replace("s-uniward", "uniw")

        actor_dir = aletheialib.utils.absolute_path(os.path.dirname(A_files[0]))
        B_dir = os.path.join(actor_dir+"-B-cache", method.upper())

        if os.path.exists(B_dir):
            print(f"Cache directory already exists:", B_dir)
        else:
            print(f"Prepare embeddings:", B_dir)
            os.makedirs(B_dir, exist_ok=True)
            aletheialib.stegosim.embed_message(fn_sim, actor_dir, "0.10-0.50", B_dir, 
                        embed_fn_saving=embed_fn_saving, show_debug_info=False)


        B_nn = aletheialib.models.NN("effnetb0")
        B_files = glob.glob(os.path.join(B_dir, '*'))

        if method in A_models:
            ops.reset_default_graph()
            A_nn = A_models[method]
            gc.collect()
        else:
            A_nn = aletheialib.models.NN("effnetb0")
            A_nn = _load_model(A_nn, "effnetb0-A-alaska2-"+method)
            A_models[method] = A_nn

        if method in B_models:
            ops.reset_default_graph()
            B_nn = B_models[method]
            gc.collect()
        else:
            B_nn = aletheialib.models.NN("effnetb0")
            B_nn = _load_model(B_nn, "effnetb0-B-alaska2-"+method)
            B_models[method] = B_nn


        # Predictions for the DCI method
        _p_aa = A_nn.predict(A_files, 50)
        _p_ab = A_nn.predict(B_files, 50)
        _p_bb = B_nn.predict(B_files, 50)
        _p_ba = B_nn.predict(A_files, 50)

        p_aa = np.round(_p_aa).astype('uint8')
        p_ab = np.round(_p_ab).astype('uint8')
        p_bb = np.round(_p_bb).astype('uint8')
        p_ba = np.round(_p_ba).astype('uint8')

        # Inconsistencies
        inc = ( (p_aa!=p_bb) | (p_ba!=0) | (p_ab!=1) ).astype('uint8')

        positives = 0
        for i in range(len(p_aa)):
            r = ""
            if inc[i]:
                r = str(round(_p_aa[i],3))+" (inc)"
            else:
                r = round(_p_aa[i],3)
            #print(A_files[i], "\t", r)
            if p_aa[i] == 1:
                positives += 1

        dci = round(1-float(np.sum(inc==1))/(2*len(p_aa)),3)
        positives_perc = round((positives)/len(p_aa), 3)
        #print(f"method={method}, DCI-pred={dci}, positives={positives_perc}")
        output += f"{dci}, {positives_perc}, "

    bn = os.path.basename(image_dir)
    with open(output_file, "a+") as myfile:
        myfile.write(bn+", "+output[:-2]+"\n")


def actor_predict_fea():
    if len(sys.argv)<4:
        print(sys.argv[0], "actor-predict-fea <actor imgage dir> <output file> [dev]\n")
        print("Example:");
        print(sys.argv[0], "actor-predict-fea actors/A1 data.csv\n")
        sys.exit(0)

    if len(sys.argv)<5:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[4]

    if os.path.exists(sys.argv[3]):
        os.remove(sys.argv[3]);
    _actor(sys.argv[2], sys.argv[3], dev_id)

def actors_predict_fea():
    if len(sys.argv)<4:
        print(sys.argv[0], "actors-predict-fea <actors dir> <output file> [dev]\n")
        print("Example:");
        print(sys.argv[0], "actors-predict-fea actors/ data.csv\n")
        sys.exit(0)

    if len(sys.argv)<5:
        dev_id = "CPU"
        print("'dev' not provided, using:", dev_id)
    else:
        dev_id = sys.argv[4]

    if os.path.exists(sys.argv[3]):
        os.remove(sys.argv[3]);
    for path in glob.glob(os.path.join(sys.argv[2], '*')):
        if not os.path.isdir(path):
            continue
        _actor(path, sys.argv[3], dev_id)



# }}}


# {{{ ats
def ats():

    import aletheialib.options

    if len(sys.argv) not in [5, 6]:
        print(sys.argv[0], "ats <embed-sim> <payload> <fea-extract> <images>")
        print(sys.argv[0], "ats <custom command> <fea-extract> <images>\n")
        print(aletheialib.options.embsim.doc)
        print("")
        print(aletheialib.options.feaext.doc)
        print("")
        print("Examples:")
        print(sys.argv[0], "ats hill-sim 0.40 srm image_dir/")
        print(sys.argv[0], "ats 'steghide embed -cf <IMAGE> -ef secret.txt -p mypass' srm image_dir/\n")
        sys.exit(0)


    import aletheialib.models
    import aletheialib.utils
    import aletheialib.stegosim
    import aletheialib.feaext

    embed_fn_saving=False

    if len(sys.argv) == 6:
        emb_sim=sys.argv[2]
        payload=sys.argv[3]
        feaextract=sys.argv[4]
        A_dir=sys.argv[5]
        fn_sim=aletheialib.stegosim.embedding_fn(emb_sim)
        fn_feaextract=aletheialib.feaext.extractor_fn(feaextract)
        if emb_sim in ["j-uniward-sim", "j-uniward-color-sim", 
                       "ued-sim", "ued-color-sim", "ebs-sim", "ebs-color-sim",
                       "nsf5-sim", "nsf5-color-sim"]:
            embed_fn_saving = True
    else:
        print("custom command")
        payload=sys.argv[2] # uggly hack
        feaextract=sys.argv[3]
        A_dir=sys.argv[4]
        fn_sim=aletheialib.stegosim.custom
        embed_fn_saving = True
        fn_feaextract=aletheialib.feaext.extractor_fn(feaextract)

    B_dir=tempfile.mkdtemp()
    C_dir=tempfile.mkdtemp()
    stegosim.embed_message(fn_sim, A_dir, payload, B_dir, embed_fn_saving=embed_fn_saving)
    stegosim.embed_message(fn_sim, B_dir, payload, C_dir, embed_fn_saving=embed_fn_saving)

    fea_dir=tempfile.mkdtemp()
    A_fea=os.path.join(fea_dir, "A.fea")
    C_fea=os.path.join(fea_dir, "C.fea")
    feaext.extract_features(fn_feaextract, A_dir, A_fea)
    feaext.extract_features(fn_feaextract, C_dir, C_fea)

    A = pandas.read_csv(A_fea, delimiter = " ").values
    C = pandas.read_csv(C_fea, delimiter = " ").values

    X=numpy.vstack((A, C))
    y=numpy.hstack(([0]*len(A), [1]*len(C)))

    from aletheialib import models
    clf=models.Ensemble4Stego()
    clf.fit(X, y)


    files=[]
    for dirpath,_,filenames in os.walk(B_dir):
        for f in filenames:
            path=os.path.abspath(os.path.join(dirpath, f))
            if not utils.is_valid_image(path):
                print("Warning, this is not a valid image: ", f)
            else:
                files.append(path)

    for f in files:
        B = fn_feaextract(f)
        B = B.reshape((1, B.shape[0]))
        p = clf.predict(B)
        if p[0] == 0:
            print(os.path.basename(f), "Cover")
        else:
            print(os.path.basename(f), "Stego")

    shutil.rmtree(B_dir)
    shutil.rmtree(C_dir)
    shutil.rmtree(fea_dir)

# }}}


