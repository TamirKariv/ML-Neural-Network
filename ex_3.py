import numpy as np
import sys

# Relu function.
ReLU = lambda x: np.maximum(0, x)


# get the inputs from the user split to training set(80%) and validation set(20%).
def get_inputs():
    train_X = np.loadtxt(sys.argv[1], dtype=int, delimiter=" ")
    train_Y = np.loadtxt(sys.argv[2], dtype=int, delimiter=" ")
    test_X = np.loadtxt(sys.argv[3], dtype=int, delimiter=" ")
    validate_X = train_X[54000:55000]
    valdiate_Y = train_Y[54000:55000]
    train_X = train_X[:54000]
    train_Y = train_Y[:54000]
    # save the seed for the next runs
    np.random.seed(0)
    rand = np.random.randint(54000, size=9)
    return train_X, train_Y, validate_X, valdiate_Y, test_X


# The Neural Network Model.
class NN:
    def __init__(self, train_X, train_Y, test_X, input_layer_size, hidden_layer_size, output_layer_size,
                 batch_size=1105, eta=0.00075, threshold=14):
        self.train_x = train_X
        self.train_y = train_Y
        self.test_x = test_X
        self.w1 = np.random.randn(input_layer_size, hidden_layer_size) * 0.0001
        self.w2 = np.random.randn(hidden_layer_size, output_layer_size) * 0.0001
        self.b1 = np.zeros((1, hidden_layer_size))
        self.b2 = np.zeros((1, output_layer_size))
        self.h1 = None
        self.g_w1 = None
        self.g_w2 = None
        self.g_b1 = None
        self.g_b2 = None
        self.eta = eta
        self.loss = 1
        self.batch_size = batch_size
        self.threshold = threshold

    # compute all the layers to the output layer.
    def forward_pass(self, train_X):
        self.h1 = ReLU(np.dot(train_X, self.w1) + self.b1)
        output = np.dot(self.h1, self.w2) + self.b2
        return output

    # calculate and update the negative log-likelihood loss.
    def update_loss(self, output, train_Y):
        max = np.max(output, axis=1, keepdims=True)
        max_exp = np.exp(output - max)
        prob = max_exp / np.sum(max_exp, axis=1, keepdims=True)
        logprob = -np.log(prob[range(train_Y.shape[0]), train_Y])
        self.loss = np.sum(logprob) / train_Y.shape[0]
        # regularize the loss.
        self.loss += 0.5 * np.sum(self.w1 * self.w1) + 0.5 * np.sum(self.w2 * self.w2)
        return prob

    # use backpropagation to find the gradients.
    def bprop(self, train_X, train_Y, prob):
        prob[range(train_Y.shape[0]), train_Y] -= 1
        prob /= train_Y.shape[0]
        self.g_b2 = np.sum(prob, axis=0, keepdims=True)
        d_w2 = np.dot(self.h1.T, prob)
        self.g_w2 = d_w2 + self.w2
        d_h1 = np.dot(prob, self.w2.T)
        d_h1[self.h1 <= 0] = 0
        self.g_b1 = np.sum(d_h1, axis=0, keepdims=True)
        d_w1 = np.dot(train_X.T, d_h1)
        self.g_w1 = d_w1 + self.w1

    # update the parameters using the gradients and learning rate.
    def update_parameters(self, w1_val, w2_val, b1_val, b2_val):
        w1_val = 0.9 * w1_val - self.eta * self.g_w1
        w2_val = 0.9 * w2_val - self.eta * self.g_w2
        b1_val = 0.9 * b1_val - self.eta * self.g_b1
        b2_val = 0.9 * b2_val - self.eta * self.g_b2
        self.w1 = self.w1 + w1_val
        self.w2 = self.w2 + w2_val
        self.b1 = self.b1 + b1_val
        self.b2 = self.b2 + b2_val
        return w1_val, w2_val, b1_val, b2_val

    # train the model with SGD.
    def train_model(self):
        w1_val, w2_val, b1_val, b2_val = 0, 0, 0, 0
        num_of_iters_in_threshold = int(self.train_x.shape[0] / self.batch_size)
        total_iters = int(num_of_iters_in_threshold * self.threshold + 1)
        for iter in range(1, total_iters):
            rand_idx = np.random.choice(self.train_x.shape[0], self.batch_size, replace=True)
            train_X_sample = self.train_x[rand_idx, :]
            train_Y_sample = self.train_y[rand_idx]
            output = self.forward_pass(train_X_sample)
            prob = self.update_loss(output, train_Y_sample)
            self.bprop(train_X_sample, train_Y_sample, prob)
            w1_val, w2_val, b1_val, b2_val = self.update_parameters(w1_val, w2_val, b1_val, b2_val)
            if iter % num_of_iters_in_threshold == 0:
                # regularize the learning rate.
                self.eta *= 0.95

    # test the accuracy of the model.
    def test_accuracy(self, test_X, test_Y):
        miss = 0
        for x, y in zip(test_X, test_Y):
            out = self.forward_pass(x)
            y_hat = np.argmax(out, axis=1)
            if y != y_hat:
                miss += 1
        return 1 - miss / test_Y.shape[0]

    # get the results from the model.
    def get_results_from_model(self):
        results = np.zeros(self.test_x.shape[0]).astype(int)
        for idx, x in enumerate(self.test_x):
            out = self.forward_pass(x)
            res = np.argmax(out, axis=1)
            results[idx] = res
        return results


# write the the results the file.
def write_results_to_file(file_name, results):
    f = open(file_name, "w")
    last_line = results.shape[0] - 1
    for idx, res in enumerate(results):
        f.write(str(res))
        if idx != last_line:
            f.write("\n")
    f.close()

def main():
    # get the inputs.
    train_X, train_Y, validate_X, validate_Y, test_X = get_inputs()
    # create the network and train it.
    nn = NN(train_X, train_Y, test_X, train_X.shape[1], 97, np.unique(train_Y).shape[0])
    nn.train_model()
    # get the predictions from the model.
    results = nn.get_results_from_model()
    # write them to the file.
    write_results_to_file("test_y",results)


if __name__ == '__main__':
    main()
