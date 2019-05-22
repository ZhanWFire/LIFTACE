function [ test_outputs, test_pre_labels, model, cost_time ] = mltest_liftex4( ds_idx,train_data,train_target,test_data,test_target, midres_mat_name )
% test liftex
    %Set the ratio parameter used by LIFT
    param.ratio = 0.1;

    % Set the kernel type used by Libsvm
    param.svm.type = 'Linear';
    param.svm.para = [];

    t0 = cputime;
    [model] = LIFTex4_train(train_data,train_target,param,midres_mat_name);
    cost_time = cputime-t0;
    model.func_name = 'liftex';
    [test_outputs,test_pre_labels,t2] = LIFTex_test(model,train_data,train_target,test_data,test_target);
end

