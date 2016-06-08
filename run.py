import sys
import random
import csv
import numpy as np

from sklearn.cross_validation import KFold

# custom modules
import network
from utils import data

FOLDS = 10

def main():
    filename = sys.argv[1]

    X = data.load_dataset('{}_X.npy'.format(filename))
    Y = data.load_dataset('{}_Y.npy'.format(filename))

    model = network.build_model()

    # vizualize the model
    network.vizualize_model(model, filename)

    # 80:20
    # print network.train_model(model, (X, Y))
    # score = model.evaluate(X, Y, verbose=0)
    # print 'Test score:', score[0]
    
    # K-Fold
    val_error = []
    losses = []
    kf = KFold(Y.shape[0], n_folds=FOLDS, shuffle=True, random_state=None)
    for train_index, val_index in kf:
        # Generate the dataset for this fold
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        print X_train.shape, X_val.shape
        print Y_train.shape, Y_val.shape

        # Train the model on this dataset
        train_history, loss_history = network.train_model(model, (X_train, Y_train), (X_val, Y_val))

        # TODO: save the losses to a file.
        losses.append(loss_history.losses)

        # Evaluate the model
        val_error = model.evaluate(X_val, Y_val, verbose=0)
        print 'Validation error:', val_error

        # NOTE: hack to run only one split
        break
        
    # Print final K-Fold error
    print "K-Fold Error: %0.2f (+/- %0.2f)" % (val_error.mean(), val_error.std() * 2)
        
    # Predict some labels
    # TODO: modify this to suit our image needs.
    counter = 0
    while counter < 1:
        idx = random.choice(xrange(Y.shape[0]))
        prediction = network.predict_model(model, np.expand_dims(X[idx,:], axis=0))
        print 'Testing: sample={}, prediction={}, actual={}'.format(
            idx, prediction, Y[idx,:])

        # save this file
        data.generate_image(prediction)
        counter += 1


    # dump the model to the file
    network.save_model(model, filename)


if __name__ == '__main__':
    main()
