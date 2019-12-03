import matplotlib.pyplot as plt


plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True


def plot_squared_conv(squared_conv, show=False):
	plt.figure()
	plt.imshow(squared_conv, extent=[0,181,181, 0])
	plt.colorbar()
	if show: plt.show()


def show():
	plt.show()