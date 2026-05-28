function [Y, scores1, scores_1] = modelPredictions(net,mbq,classes, net1)

Y = [];
scores1 = [];

% Loop over mini-batches.
while hasdata(mbq)
    X = next(mbq);

    % Make prediction.
    scores = predict(net,X);
    scores_1 = predict(net1,X);

    % Decode labels and append to output.
    labels = onehotdecode(scores,classes,1)';
    Y = [Y; labels];
    if length(scores_1) == 128
    scores1 = [scores1, scores_1];
    end
    
end

end