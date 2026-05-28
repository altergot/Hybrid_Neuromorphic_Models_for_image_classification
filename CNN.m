tic
close all
clearvars

num_runs = 10;
num_epochs_total = 50;
noise_level = 0.15;
miniBatchSize = 128;
initialLearnRate = 0.01;
decay = 0.001;
momentum = 0.9;
N1 = 200;
N2 = 100;
N3 = 10;
inputSize = [28 28 1];
pixelRange = [-5 5];

razm_class      = zeros(10, num_epochs_total+1, num_runs);
razm_class_n    = zeros(10, num_epochs_total+1, num_runs);
rasst_m_class   = zeros(10, 10, num_epochs_total+1, num_runs);
rasst_m_class_n = zeros(10, 10, num_epochs_total+1, num_runs);
accur           = zeros(num_epochs_total+1, num_runs);


for run = 1:num_runs
    fprintf('\n===== Run %d =====\n', run);

    rng(42 + run - 1);


    digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
        'nndatasets','DigitDataset');
    imds = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');

    numTrainFiles = 750;
    [imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize');

    for i = 1:2500
        I = string(imdsValidation.Files(i,1));
        image = imread(I);
        image = imresize(image, [81 81]);
        image(image < 50) = 0;
        image(image > 0) = 1;
        image = double(image);

        if noise_level > 0
            num = 81 * 81;
            p = randperm(num);
            ind = p(1:uint16(num * noise_level));
            image(ind) = abs(image(ind) - 1);
        end

        image = uint8(image);
        image(image > 0) = 255;
        image = imresize(image, [28 28], 'nearest');

        old = 'C:\Program Files\MATLAB\R2023b\toolbox\nnet\nndemos\nndatasets';
        new = 'C:\Digit1';
        I_new = replace(I, old, new);
        imwrite(image, I_new);
        imdsValidation.Files(i,1) = cellstr(I_new);
    end

    for i = 1:7500
        I = string(imdsTrain.Files(i,1));
        image = imread(I);
        image = imresize(image, [81 81]);
        image(image < 50) = 0;
        image(image > 0) = 1;
        image = double(image);

        image = uint8(image);
        image(image > 0) = 255;
        image = imresize(image, [28 28], 'nearest');

        old = 'C:\Program Files\MATLAB\R2023b\toolbox\nnet\nndemos\nndatasets';
        new = 'C:\Digit1';
        I_new = replace(I, old, new);
        imwrite(image, I_new);
        imdsTrain.Files(i,1) = cellstr(I_new);
    end

    imageAugmenter = imageDataAugmenter( ...
        'RandXTranslation', pixelRange, ...
        'RandYTranslation', pixelRange);

    augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
        'DataAugmentation', imageAugmenter);
    augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

    mbqMean = minibatchqueue(augimdsTrain, ...
        'MiniBatchSize', miniBatchSize, ...
        'MiniBatchFcn', @preprocessMiniBatch, ...
        'MiniBatchFormat', {'SSCB',''});
    images_for_training = [];
    shuffle(mbqMean);
    while hasdata(mbqMean)
        dlX = next(mbqMean);
        images_for_training = cat(4, images_for_training, dlX);
    end
    mean_image = mean(images_for_training, 4);
    mean_image = cat(4, [], extractdata(mean_image));

    layers = [
        imageInputLayer([28 28 1], 'Name', 'input', 'Mean', mean_image)
        convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv_1')
        batchNormalizationLayer('Name', 'BN_1')
        reluLayer('Name', 'relu_1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'max_pool_1')

        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_2')
        batchNormalizationLayer('Name', 'BN_2')
        reluLayer('Name', 'relu_2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'max_pool_2')

        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_3')
        batchNormalizationLayer('Name', 'BN_3')
        reluLayer('Name', 'relu_3')

        fullyConnectedLayer(N1, 'Name', 'FC_buffer')
        reluLayer('Name', 'FC_buffer_relu_out')

        fullyConnectedLayer(N2, 'Name', 'FC_attack')
        reluLayer('Name', 'relu_5')

        fullyConnectedLayer(N3, 'Name', 'FC_1')
        softmaxLayer('Name', 'soft_max_1')
        ];

    lgraph = layerGraph(layers);
    dlnet = dlnetwork(lgraph);

    velocity = [];

    mbq = minibatchqueue(augimdsTrain, ...
        'MiniBatchSize', miniBatchSize, ...
        'MiniBatchFcn', @preprocessMiniBatch, ...
        'MiniBatchFormat', {'SSCB',''});

    classes = categories(imdsTrain.Labels);

    fprintf('Epoch 0 (до обучения)\n');

    lgraph_eval = layerGraph(dlnet);
    lgraph_eval = removeLayers(lgraph_eval, 'soft_max_1');
    lgraph_embed = removeLayers(lgraph_eval, 'FC_1');
    dlnet_embed = dlnetwork(lgraph_embed);

    mbqTest = minibatchqueue(augimdsValidation, 1, ...
        'MiniBatchSize', miniBatchSize, ...
        'MiniBatchFcn', @preprocessMiniBatchPredictors, ...
        'MiniBatchFormat', 'SSCB');

    [predictions, X_test, X_last_tests] = modelPredictions(dlnet, mbqTest, classes, dlnet_embed);
    YTest = imdsValidation.Labels;
    accuracy = mean(predictions == YTest);

    X_embed = [X_test, X_last_tests];
    embeddings = extractdata(gather(X_embed));
    embeddings = squeeze(embeddings)';
    embeddings = double(embeddings);
    labels = double(grp2idx(YTest));

    unique_classes = unique(labels);
    num_classes = length(unique_classes);
    class_centroids = zeros(num_classes, size(embeddings,2));
    class_sizes = zeros(num_classes, 1);
    for i = 1:num_classes
        idx = (labels == unique_classes(i));
        class_emb = embeddings(idx, :);
        class_centroids(i,:) = mean(class_emb, 1);
        centroid = class_centroids(i,:);
        distances = sqrt(sum((class_emb - centroid).^2, 2));
        class_sizes(i) = mean(distances);
    end

    euclidean_dist = squareform(pdist(class_centroids, 'euclidean'));

    mean_all = mean(embeddings, 1);
    embeddings_centered = embeddings - mean_all;
    cov_matrix = cov(embeddings_centered);
    [~, eigenvalues] = eig(cov_matrix);
    eigenvalues = diag(eigenvalues);
    eigenvalues = sort(eigenvalues, 'descend');
    total_variance = sum(eigenvalues);
    scale_factor = sqrt(total_variance);
    euclidean_dist_norm = euclidean_dist / scale_factor;
    class_sizes_norm = class_sizes / scale_factor;

    razm_class(:, 1, run) = class_sizes;
    razm_class_n(:, 1, run) = class_sizes_norm;
    rasst_m_class(:, :, 1, run) = euclidean_dist;
    rasst_m_class_n(:, :, 1, run) = euclidean_dist_norm;
    accur(1, run) = accuracy;

    iteration = 0;
    start = tic;

    for epoch = 1:num_epochs_total
        shuffle(mbq);
        while hasdata(mbq)
            iteration = iteration + 1;
            [dlX, dlY] = next(mbq);
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
            [gradients, state, loss] = dlfeval(@modelGradients, dlnet, dlX, dlY);
            dlnet.State = state;
            learnRate = initialLearnRate / (1 + decay * iteration);
            [dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        end

        fprintf('Epoch %d\n', epoch);

        lgraph_eval = layerGraph(dlnet);
        lgraph_eval = removeLayers(lgraph_eval, 'soft_max_1');
        lgraph_embed = removeLayers(lgraph_eval, 'FC_1');
        dlnet_embed = dlnetwork(lgraph_embed);

        mbqTest = minibatchqueue(augimdsValidation, 1, ...
            'MiniBatchSize', miniBatchSize, ...
            'MiniBatchFcn', @preprocessMiniBatchPredictors, ...
            'MiniBatchFormat', 'SSCB');

        [predictions, X_test, X_last_tests] = modelPredictions(dlnet, mbqTest, classes, dlnet_embed);
        YTest = imdsValidation.Labels;
        accuracy = mean(predictions == YTest);

        X_embed = [X_test, X_last_tests];
        embeddings = extractdata(gather(X_embed));
        embeddings = squeeze(embeddings)';
        embeddings = double(embeddings);
        labels = double(grp2idx(YTest));

        unique_classes = unique(labels);
        num_classes = length(unique_classes);
        class_centroids = zeros(num_classes, size(embeddings,2));
        class_sizes = zeros(num_classes, 1);
        for i = 1:num_classes
            idx = (labels == unique_classes(i));
            class_emb = embeddings(idx, :);
            class_centroids(i,:) = mean(class_emb, 1);
            centroid = class_centroids(i,:);
            distances = sqrt(sum((class_emb - centroid).^2, 2));
            class_sizes(i) = mean(distances);
        end

        euclidean_dist = squareform(pdist(class_centroids, 'euclidean'));

        mean_all = mean(embeddings, 1);
        embeddings_centered = embeddings - mean_all;
        cov_matrix = cov(embeddings_centered);
        [~, eigenvalues] = eig(cov_matrix);
        eigenvalues = diag(eigenvalues);
        eigenvalues = sort(eigenvalues, 'descend');
        total_variance = sum(eigenvalues);
        scale_factor = sqrt(total_variance);
        euclidean_dist_norm = euclidean_dist / scale_factor;
        class_sizes_norm = class_sizes / scale_factor;

        razm_class(:, epoch+1, run) = class_sizes;
        razm_class_n(:, epoch+1, run) = class_sizes_norm;
        rasst_m_class(:, :, epoch+1, run) = euclidean_dist;
        rasst_m_class_n(:, :, epoch+1, run) = euclidean_dist_norm;
        accur(epoch+1, run) = accuracy;

        save('DATA.mat', 'razm_class', 'razm_class_n', ...
            'rasst_m_class', 'rasst_m_class_n', 'accur', '-v7.3');
    end

    toc
end