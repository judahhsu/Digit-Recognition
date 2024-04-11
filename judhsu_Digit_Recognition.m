% Darling Judah Hsu, SID: 921108366

%% Step 02
% Display First 16 images of train_patterns
load('USPS.mat')
figure;
for i=1:16
    image = train_patterns(:, i);
    image_shaped = reshape(image, [16, 16]);
    image_shaped = image_shaped';
    subplot(4, 4, i);
    imagesc(image_shaped);
    colormap(gray);
end

% Computing mean digits
figure;
train_aves = [];
for k=1:10
    digits = train_patterns(:, train_labels(k,:)==1);
    digits_ave = mean(digits, 2);
    train_aves = [train_aves, digits_ave];
end
for i=1:10
    image = train_aves(:, i);
    image_shaped = reshape(image, [16, 16]);
    image_shaped = image_shaped';
    subplot(4, 4, i);
    imagesc(image_shaped);
    colormap(gray);
end

%% Step 03

% a) Calculate Euclidean Distance between test and mean
test_classif = [];
for k=1:10
    digit_dif = sum((test_patterns-repmat(train_aves(:,k),[1 4649])).^2);
    test_classif = [test_classif;digit_dif];
end

% b) Getting index of k with smallest error
test_classif_res = [];
for j=1:4649
    [tmp, ind] = min(test_classif(:,j));
    test_classif_res = [test_classif_res ind];
end
    
% c) Centroid confusion matrix
test_confusion = [];
for k=1:10
    tmp=test_classif_res(test_labels(k,:)==1);
    row_k = [];
    for j=1:10
        count = sum(tmp == j);
        row_k = [row_k count];
    end
    test_confusion = [test_confusion; row_k];
end

%% Step 04

% a) Find SVDs
train_u = [];
for k=1:10
    [train_u(:,:,k),tmp,tmp2] = svds(train_patterns(:,train_labels(k,:)==1),17);
end

% b) Expansion coefficients
test_svd17 = [];
for k=1:10
    test_svd17(:,:,k) = train_u(:,:,k)' * test_patterns;
end

% c) find residual errors of SVD approximations for k:0-9
test_svd17res = [];
for k=1:10
    approx_mat = train_u(:,:,k)*test_svd17(:,:,k);
    difference_mat = test_patterns - approx_mat;
    error_vec = [];
    for j=1:4649
        error_vec = [error_vec norm(difference_mat(:,j),2)];
    end
    test_svd17res = [test_svd17res; error_vec];
end

% d) Find smallest error & compute confusion matrix again
test_svd17_min = [];
for j=1:4649
    [tmp, ind] = min(test_svd17res(:,j));
    test_svd17_min = [test_svd17_min ind];
end
test_svd17_confusion = [];
for k=1:10
    tmp=test_svd17_min(test_labels(k,:)==1);
    row_k = [];
    for j=1:10
        count = sum(tmp == j);
        row_k = [row_k count];
    end
    test_svd17_confusion = [test_svd17_confusion; row_k];
end