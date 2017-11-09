import numpy as np
import matplotlib.pyplot as plt
import itertools


# Step 1
def load_data(input_path, output_path):
    x = np.loadtxt(input_path)
    y = np.loadtxt(output_path)
    return x, y

x, y = load_data("dataset1_inputs.txt", "dataset1_outputs.txt")

plt.plot(x, y, "x", color="green")
plt.show()

# Step 2


def poly(i, x):
    return x ** (i - 1)


def generate_phi(D, x):
    phi = [[0] * D for i in range(len(x))]
    for i, j in enumerate(x):
        phi[i] = [poly(d, j) for d in range(1, D + 1)]
    return phi


def generate_w_mle(phi, y):
    inner = np.dot(np.transpose(phi), phi)
    outer = np.dot(np.linalg.inv(inner), np.transpose(phi))
    w = np.dot(outer, y)
    return w


def compute_polynomials(w, x, d):
        total = 0
        for i in range(d):
            total += w[i] * poly(i+1, x)
        return total


def compute_mse(D, w, y, x):

    mse = 0
    for j in range(len(y)):
        mse += (y[j] - compute_polynomials(w, x[j], D))**2
    return mse / len(y)


num_basis = 20
mses = np.zeros(num_basis)
for D in range(1, num_basis + 1):
    phi = generate_phi(D, x)
    w = generate_w_mle(phi, y)
    mses[D - 1] = compute_mse(D, w, y, x)

print mses
print np.argmin(mses) + 1
plt.plot(range(0, 21), np.concatenate(([None], mses)))
plt.title("MSE as a Function of D with MLE Estimate")
plt.xlabel("D")
plt.ylabel("MSE")
plt.xlim(0, 21)
plt.show()


# Step 3
def generate_w_map(phi, y, d):
    lam = 0.001
    identity = lam * np.identity(d)
    phi_t = np.dot(np.transpose(phi), phi)
    inner = np.add(identity, phi_t)
    outer = np.dot(np.linalg.inv(inner), np.transpose(phi))
    w = np.dot(outer, y)
    return w

mses_mle = mses
num_basis = 20
mses = np.zeros(num_basis)
for D in range(1, num_basis + 1):
    phi = generate_phi(D, x)
    w = generate_w_map(phi, y, D)
    mses[D - 1] = compute_mse(D, w, y, x)

print mses
print np.argmin(mses) + 1
plt.plot(range(0, 21), np.concatenate(([None], mses)), label="MAP", color="blue")
plt.plot(range(0, 21), np.concatenate(([None], mses_mle)), label="MLE", color="red")
plt.legend()
plt.title("MSE as a Function of D with MAP and MLE Estimate")
plt.xlabel("D")
plt.ylabel("MSE")
plt.xlim(0, 21)
plt.show()

# Step 4


for D in range(1, num_basis + 1):
    phi = generate_phi(D, x)
    w_mle = generate_w_mle(phi, y)
    w_map = generate_w_map(phi, y, D)

    x_new = np.linspace(min(x), max(x), 200)     # Create new x range for sampling more points

    pred_y_mle = np.zeros(len(x_new))
    pred_y_map = np.zeros(len(x_new))
    for i in range(len(x_new)):
        pred_y_mle[i] = compute_polynomials(w_mle, x_new[i], D)
        pred_y_map[i] = compute_polynomials(w_map, x_new[i], D)

    data_mle = sorted(itertools.izip(*[x_new, pred_y_mle]))
    mle_x, mle_y = list(itertools.izip(*data_mle))
    plt.plot(mle_x, mle_y, "-", label="MLE", color="red")

    data_map = sorted(itertools.izip(*[x_new, pred_y_map]))
    map_x, map_y = list(itertools.izip(*data_map))
    plt.plot(map_x, map_y, "-", label="MAP", color="blue")

    plt.plot(x, y, "x", label="Data", color="green")
    plt.title("D = {}".format(D))
    plt.legend()
    plt.show()


# Step 5

fold = 10
avg = np.zeros(num_basis)

for i in range(1, num_basis + 1):
    mses = np.zeros(10)
    xy = np.random.permutation(zip(x, y))
    new_x = xy[:, 0]
    new_y = xy[:, 1]
    for j in range(0, len(new_x), fold):

        training = np.concatenate([new_x[0:j], new_x[j+fold : len(new_x)]])
        training_y = np.concatenate([new_y[0:j], new_y[j+fold: len(new_y)]])

        validation = new_x[j:j + fold]
        valid_y = new_y[j:j + fold]

        phi = generate_phi(i, training)
        w_map = generate_w_map(phi, training_y, i)

        mses[j/fold] = compute_mse(i, w_map, valid_y, validation)

    avg[i-1] = np.mean(mses)

print np.argmin(avg)
plt.plot(range(0, 21), np.concatenate(([None], avg)))
plt.title("Ten Fold Cross Validation Average MSE results")
plt.xlabel("D")
plt.ylabel("MSE")
plt.show()

# Step 6

x, y = load_data("dataset2_inputs.txt", "dataset2_outputs.txt")
plt.plot(x, y, "x", color="green")
plt.show()


# Step 7

tau = 1000
sigma = 1.5
d = 13

phi = generate_phi(d, x)
ide = tau**(-2) * np.identity(d)
sig = sigma**(-2) * np.dot(np.transpose(phi), phi)
inv = np.add(ide, sig)
cov_w = np.linalg.inv(inv)
mu_w = sigma**(-2) * np.dot(np.dot(cov_w, np.transpose(phi)), y)

new_x = np.linspace(min(x), max(x), 200)

mu_d = np.zeros(len(new_x))
sig_d = np.zeros(len(new_x))
plus_sig = np.zeros(len(new_x))
minus_sig = np.zeros(len(new_x))

for i in range(len(new_x)):

    phi_x = [poly(D, new_x[i]) for D in range(1, d + 1)]

    mu_d[i] = np.dot(np.transpose(mu_w), phi_x)
    sig_d[i] = np.sqrt(sigma**2 + np.dot(np.dot(np.transpose(phi_x), cov_w), phi_x))
    plus_sig[i] = mu_d[i] + sig_d[i]
    minus_sig[i] = mu_d[i] - sig_d[i]

plt.plot(x, y, "x", color="green")

data_mu = sorted(itertools.izip(*[new_x, mu_d]))
mu_x, mu_y = list(itertools.izip(*data_mu))
plt.plot(mu_x, mu_y, "-", color="blue")

data_plus = sorted(itertools.izip(*[new_x, plus_sig]))
plus_x, plus_y = list(itertools.izip(*data_plus))
plt.plot(plus_x, plus_y, "--", color="green")

data_minus = sorted(itertools.izip(*[new_x, minus_sig]))
minus_x, minus_y = list(itertools.izip(*data_minus))
plt.plot(minus_x, minus_y, "--", color="green")
plt.title("D = {}".format(d))
plt.show()


data_sig = sorted(itertools.izip(*[new_x, sig_d]))
sig_x, sig_y = list(itertools.izip(*data_sig))
plt.plot(sig_x, sig_y)
plt.title("Standard Deviation as a function of x")
plt.xlabel("x")
plt.ylabel("Standard Deviation")
plt.show()
