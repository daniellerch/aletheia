function ensemble_fit(path_train_cover, path_train_stego, path_clf_out)

    TRN_cover = load(path_train_cover);
    TRN_stego = load(path_train_stego);

    % We need the same number of samples in both classes
    if size(TRN_cover.F,1)>size(TRN_stego.F,1)
        TRN_cover.F=TRN_cover.F(1:size(TRN_stego.F,1),:);
    end

    if size(TRN_cover.F,1)<size(TRN_stego.F,1)
        TRN_stego.F=TRN_stego.F(1:size(TRN_cover.F,1),:);
    end

    % set PRNG initialization
    %RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
    rand('state', 1);

    % Train ensemble with all settings default
    [clf,results] = ensemble_training(TRN_cover.F, TRN_stego.F);

    %--save(path_clf_out, 'clf');
    save('-mat7-binary', path_clf_out, 'clf');
end


