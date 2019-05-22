function [model,t1]=LIFTex4_train(train_data,train_target,param,midres_mat_name)
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

%% nargin returns the number of input arguments passed in a call to the currently executing function. Use this nargin syntax only in the body of a function
    disp(['function LIFTex is called...']);

    if(nargin<2)
        error('Not enough input parameters, please type "help LIFTex" for more information');
    end
    if nargin<3
        param = struct();
    end
    is_save_mid = (nargin>=4);
    
    param = setParam(param);

    [num_train,dim]=size(train_data);
    [num_class,~]=size(train_target);
%     is_big = true;
    is_big = ( num_train^2*num_class >= 2e9 );


    %% Find key instances of each label
    tic
    if is_save_mid && exist(midres_mat_name,'file')
        data = load(midres_mat_name,'C_IDXs','iter_1');
        C_IDXs = data.C_IDXs;
        start_1 = data.iter_1;
        clear data;
        disp(['phase1 restart from ', num2str(start_1), '-th class']);
    else
        C_IDXs = zeros(num_class,num_train);
        start_1 = 1;
    end
    for i=start_1:num_class
        disp(['Performing clusteirng for the ',num2str(i),'/',num2str(num_class),'-th class']);

        p_idx = train_target(i,:)==1;
        n_idx = train_target(i,:)~=1;
        p_num = sum(p_idx);
        n_num = sum(n_idx);
%         p_data = train_data(p_idx,:);
%         n_data = train_data(n_idx,:);
%         p_num = size(p_data,1);
%         n_num = size(n_data,1);

        k1=min(ceil(p_num*param.ratio),ceil(n_num*param.ratio));   %ceil(A)--rounds the elements of A to the nearest integers greater than or equal to A
        k2=k1;
        
        if(k1==0)
            POS_IDX = [];
            [NEG_IDX,~]=kmeans(train_data,min(50,num_train),'EmptyAction','singleton','OnlinePhase','off','Display','off');
        else
            if(p_num==1)
                POS_IDX = [1];
            else
                [POS_IDX,~]=kmeans(train_data(p_idx,:),k1,'EmptyAction','singleton','OnlinePhase','off','Display','off');
            end

            if(n_num==1)
                NEG_IDX = [1];
            else
                [NEG_IDX,~]=kmeans(train_data(n_idx,:),k2,'EmptyAction','singleton','OnlinePhase','off','Display','off');
            end
        end

        C_IDXs(i,p_idx) = POS_IDX;
        C_IDXs(i,n_idx) = NEG_IDX + k1;
        iter_1 = i+1;
        if is_save_mid
            save(midres_mat_name,'C_IDXs','iter_1','-v7.3');
        end
    end
    %%
    cos_dist = squareform(pdist(train_target, 'cosine'));
%     WC = 1-cos_dist;
%     WC(WC<0) = 0;
    WC = exp( - param.lambda*cos_dist );
    
    if ~is_big
        WIC = cell(1,num_class);
    end
    
    if is_save_mid && exist(midres_mat_name,'file') && ~isempty(whos('-file',midres_mat_name,'data_type_func'))
        data = load(midres_mat_name,'data_type_func');
        data_type_func = data.data_type_func;
        clear data;
    else
        data_type_func = @sparse;
    end
    
    for i=1:num_class
        disp(['Compute weight for the ',num2str(i),'/',num2str(num_class),'-th class']);
        WC(i,:) = WC(i,:)/sum(WC(i,:));
        if ~is_big
            idx = data_type_func(C_IDXs(i,:));
%             idx = C_IDXs(i,:);
            WIC{i} = bsxfun(@eq, idx, idx');
        else
            if is_save_mid
                w_name = ['WIC_',num2str(i)];
                if ~isempty(whos('-file',midres_mat_name,w_name))
                    disp('already saved, skip');
                    continue;
                end
                idx = data_type_func(C_IDXs(i,:));
%                 idx = C_IDXs(i,:);
                eval([w_name,' = bsxfun(@eq, idx, idx'');']);
                if 1==i
                    eval(['tmp=full(',w_name,');']);
                    info_s = whos(w_name);
                    info_f = whos('tmp');
                    if info_f.bytes < info_s.bytes
                        data_type_func = @full;
                        eval([w_name,' = tmp;']);
                    end
                    clear info_s info_f tmp;
                end
                save(midres_mat_name,w_name,'data_type_func','-append');
                eval(['clear ',w_name,';']);
            end
        end
    end
    %%
    P_Centers=cell(num_class,1);
    N_Centers=cell(num_class,1);
    C_IDXs2 = zeros(num_class,num_train);
    
    if is_save_mid && exist(midres_mat_name,'file') && ~isempty(whos('-file',midres_mat_name,'iter_2'))
        data = load(midres_mat_name,'C_IDXs2','iter_2','P_Centers','N_Centers');
        C_IDXs2 = data.C_IDXs2;
        start_2 = data.iter_2;
        P_Centers = data.P_Centers;
        N_Centers = data.N_Centers;
        clear data;
        disp(['phase2 restart from ', num2str(start_2), '-th class']);
    else
        start_2 = 1;
    end
    
    for i=start_2:num_class
        disp(['Performing ensemble clusteirng for the ',num2str(i),'/',num2str(num_class),'-th class']);
        
        p_idx = train_target(i,:)==1;
        n_idx = train_target(i,:)~=1;
        p_num = sum(p_idx);
        n_num = sum(n_idx);
        
        WIp = zeros(p_num, p_num);
        WIn = zeros(n_num, n_num);
        for j=1:num_class
            if is_big
                w_name = ['WIC_',num2str(j)];
                if is_save_mid && ~isempty(whos('-file',midres_mat_name,w_name))
                    data = load(midres_mat_name,w_name);
                    eval(['WICp=data.',w_name,'(p_idx,p_idx);']);
                    eval(['WICn=data.',w_name,'(n_idx,n_idx);']);
                    clear data;
                else
                    idx = data_type_func(C_IDXs(j,:));
%                     idx = C_IDXs(j,:);
                    WICp = bsxfun(@eq, idx(p_idx), idx(p_idx)');
                    WICn = bsxfun(@eq, idx(n_idx), idx(n_idx)');
                end
            else
                WICp = WIC{j}(p_idx,p_idx);
                WICn = WIC{j}(n_idx,n_idx);
            end
            WIp = WIp + WC(i,j)*WICp;
            WIn = WIn + WC(i,j)*WICn;
        end
        clear WICp WICn;

        k1 = min(ceil(p_num*param.ratio),ceil(n_num*param.ratio));   %ceil(A)--rounds the elements of A to the nearest integers greater than or equal to A
        k2=k1;
        
        if k1==0
            POS_IDX = [];
            k2 = min(50, num_train);
        else
            POS_IDX = cluster_on_weight(WIp, k1);
        end
        NEG_IDX = cluster_on_weight(WIn, k2);
        clear WIp WIn;
        
        POS_C = zeros(k1,dim);
        NEG_C = zeros(k2,dim);
        for j=1:k1
            cluster_idx = p_idx;
            cluster_idx(cluster_idx==1) = (POS_IDX==j);
            POS_C(j,:)=mean(train_data(cluster_idx,:));
        end
        for j=1:k2
            cluster_idx = n_idx;
            cluster_idx(cluster_idx==1) = (NEG_IDX==j);
            NEG_C(j,:)=mean(train_data(cluster_idx,:));
        end
        P_Centers{i,1}=POS_C;
        N_Centers{i,1}=NEG_C;
        C_IDXs2(i,p_idx) = POS_IDX;
        C_IDXs2(i,n_idx) = NEG_IDX + k1;
        iter_2 = i + 1;
        if is_save_mid
            save(midres_mat_name,'C_IDXs2','iter_2','P_Centers','N_Centers','-append');
        end
    end

    %%
    clearvars -except param num_class P_Centers N_Centers train_data train_target;
    Models=cell(num_class,1);
    %Perform representation transformation and training
    for i=1:num_class
        disp(['Building classifiers: ',num2str(i),'/',num2str(num_class)]);

        centers=[P_Centers{i,1};N_Centers{i,1}];
        num_center=size(centers,1);

        if(num_center>=5000)
            error('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...');
        end

        training_instance_matrix=pdist2(train_data,centers);
        training_label_vector=train_target(i,:)';
        Models{i,1}=svmtrain(training_label_vector,training_instance_matrix,param.svm.str);
    end
    t1 = toc;
    
    model.Models = Models;
    model.P_Centers = P_Centers;
    model.N_Centers = N_Centers;
end

function [param] = setParam(param)
%%
    if(~isfield(param,'lambda'))
        param.lambda = 10;
    end
    
    if(~isfield(param,'svm'))
        param.svm.type='Linear';
        param.svm.para=[];
    end

    if(~isfield(param,'ratio'))
        param.ratio=0.1;
    end
    
    switch param.svm.type
        case 'RBF'
            gamma=num2str(param.svm.para);
            param.svm.str=['-t 2 -g ',gamma,' -b 1'];
        case 'Poly'
            gamma=num2str(param.svm.para(1));
            coef=num2str(param.svm.para(2));
            degree=num2str(param.svm.para(3));
            param.svm.str=['-t 1 ','-g ',gamma,' -r ', coef,' -d ',degree,' -b 1'];
        case 'Linear'
            param.svm.str='-t 0 -b 1';
        otherwise
            error('SVM types not supported, please type "help LIFT" for more information');
    end
end

function [IDX] = cluster_on_weight(weight, k)
%%
    if k==1
        IDX = [1];
    else
        weight = weight*diag(1./sqrt(sum(weight.*weight)));
        [u,~,~] = svds(weight,k);
        u = u./repmat(sqrt(sum(u.^2,2)), 1, k);
        IDX = kmeans(u,k,'EmptyAction','singleton','OnlinePhase','off','Display','off');
    end
end
