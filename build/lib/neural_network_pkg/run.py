import extract as ext
import neural_net as nn
import csv
import numpy as np
import sys
from sklearn import preprocessing as pp
import neural_net2 as nn2

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
            
def getdata_file(filename):
    with open(filename,'r') as f:
        creader = csv.reader(f)
        data = np.array(list(creader)).astype(np.float)
    np.random.shuffle(data)
    x = data[:,:-4]
    y = data[:,-4:]
    ylabel = list()
    for ydata in y:
        for l in range(len(ydata)):
            if ydata[l]==1:
                ylabel.append(l)
                break
    return x, y, ylabel


def main():
    #Extract data from images
    target_dir = '../../Dataset/'
    if extract_data_again:
        ext.main()
    xtrain, ytrain, ytrain_label = getdata_file(target_dir+'DS2full-train.csv')
    xtest, ytest, ytest_label = getdata_file(target_dir+'DS2full-test.csv')

    #Scale Data
    scaler = pp.MinMaxScaler()
    scaler.fit(np.concatenate((xtrain,xtest)))
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)
    original = sys.stdout
    eta = 0.1
    
    filelog=open('result_'+str(eta)+'.txt','w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, filelog)
        
    print('##########################################')
    print('Eta:',eta)
    n_epoch = 80
    no_hidden = 30
    NeuralNet = nn.neural_network(len(xtrain[0]),[30],len(ytrain[0]))
    NeuralNet.train(xtrain, ytrain,ytrain_label,eta, n_epoch,test_while_train = True, xtest = xtest,ytest_label= ytest_label, draw_plot=True)
    eta = eta*10
    sys.stdout = original
    
def main2():
    #Extract data from images
    target_dir = '../../Dataset/'
    if extract_data_again:
        ext.main()
    xtrain, ytrain, ytrain_label = getdata_file(target_dir+'DS2full-train.csv')
    xtest, ytest, ytest_label = getdata_file(target_dir+'DS2full-test.csv')
    #Scale Data
    scaler = pp.MinMaxScaler()
    scaler.fit(np.concatenate((xtrain,xtest)))
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)
    
    
    original = sys.stdout
    eta = 0.01
    gamma = 0.01
    for i in range(5):
        filelog=open('result2_'+str(gamma)+'.txt','w')
        original = sys.stdout
        sys.stdout = Tee(sys.stdout, filelog)
            
        print('##########################################')
        print('Eta:',eta)
        print('Gamma:',gamma)
        #Set number of epochs
        n_epoch = 250

        no_hidden = 20
        NeuralNet = nn2.neural_network(len(xtrain[0]),[20],len(ytrain[0]))
        NeuralNet.train(xtrain, ytrain,ytrain_label,eta,gamma, n_epoch,test_while_train = True, xtest = xtest,ytest_label= ytest_label, draw_plot=True)
        sys.stdout = original
        gamma = gamma*10.0
    

if __name__ == '__main__':
    extract_data_again=False
    original = sys.stdout
    #For normal neural net
    main()
    #For regularized neural net
    main2()
