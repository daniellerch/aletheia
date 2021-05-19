function ensemble_predict(path_clf, path_test, path_votes_out)

    test = load(path_test);

    data=load(path_clf); 
    clf=data.clf;

    f=fopen(path_votes_out, 'w');
    if size(test.F, 1)>0
        results = ensemble_testing(test.F, clf);
        for idx=1:numel(results.votes_)
            vote = results.votes_(idx);
            fprintf(f, '%i\n', vote);
        end
    end
    fclose(f);

end

