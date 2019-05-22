function [Outputs,Pre_Labels,t2]=LIFTex_test(model,train_data,train_target,test_data,test_target)
%LIFT deals with multi-label learning problem by introducing label-specific features [1].
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=LIFT(train_data,train_target,test_data,test_target,ratio,svm)
%
%    Description
%
%       LIFT takes,
%           train_data       - An M1xN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target     - A QxM1 array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           test_data        - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM2 array, if the ith testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           ratio            - The number of clusters (i.e. k1 for positive examples, k2 for negative examples) considered for the i-th class is set to
%                              k2=k1=min(ratio*num_pi,ratio*num_ni), where num_pi and num_ni are the number of positive and negative examples for the i-th class respectively.
%                              ***The default configuration is ratio=0.1***
%           svm              - A struct variable with two fields, i.e. svm.type and svm.para.
%                              Specifically, svm.type gives the kernel type, which can take the value of 'RBF', 'Poly', 'Linear' or 'LibLinear';
%                              svm.para gives the corresponding parameters used for the specified kernel:
%                              1) if svm.type is 'RBF', then svm.para gives the value of gamma, where the kernel is exp(-gamma*|x(i)-x(j)|^2)
%                              2) if svm.type is 'Poly', then svm.para(1:3) gives the value of gamma, coefficient, and degree respectively, where the kernel is (gamma*<x(i),x(j)>+coefficient)^degree.
%                              3) if svm.type is 'Linear', then svm.para is [].
%                              *** The default configuration of svm is svm.type='Linear' with 'svm.para=[]' ***
%
%      and returns,
%           Outputs          - The output of the ith testing instance on the jth class is stored in Outputs(j,i)
%           Pre_Labels       - If the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1
%
%  [1] M.-L. Zhang. LIFT: Multi-label learning with label-specific features, In: Proceedings of the 22nd International Joint Conference on Artificial Intelligence, 2011.
%
%  [2] R. E. Schapire, Y. Singer. BoosTexter: A boosting based system for text categorization. Machine Learning, 39(2/3): 135-168, 2000.

%nargin returns the number of input arguments passed in a call to the currently executing function. Use this nargin syntax only in the body of a function
    Models = model.Models;
    P_Centers = model.P_Centers;
    N_Centers = model.N_Centers;

    [num_class,num_test]=size(test_target);
    %Perform representation transformation and testing
    tic;
    Pre_Labels = zeros(num_class,num_test);
    Outputs = zeros(num_class,num_test);
    for i=1:num_class
        centers=[P_Centers{i,1};N_Centers{i,1}];
        num_center=size(centers,1);

        if(num_center>=5000)
            error('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...');
        end

        testing_instance_matrix=pdist2(test_data,centers);
        testing_label_vector=test_target(i,:)';

        [predicted_label,accuracy,prob_estimates]=svmpredict(testing_label_vector,testing_instance_matrix,Models{i,1},'-b 1');
        if(isempty(predicted_label))
            predicted_label=train_target(i,1)*ones(num_test,1);
            if(train_target(i,1)==1)
                Prob_pos=ones(num_test,1);
            else
                Prob_pos=zeros(num_test,1);
            end
        else
%             pos_index=find(Models{i,1}.Label==1);
            Prob_pos=prob_estimates(:,Models{i,1}.Label==1);
        end
        Outputs(i,:)=Prob_pos';
        Pre_Labels(i,:) = predicted_label';
    end

    t2 = toc;
end