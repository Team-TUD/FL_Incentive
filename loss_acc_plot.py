import matplotlib.pyplot as plt
def loss_acc_plot(h, epochs_init, epochs_training, ifsaving, isgoogle):
    nb_epoch = epochs_init + epochs_training
    if isgoogle:
        acc, loss, val_acc, val_loss = h['acc'], h['loss'], h['val_acc'], h['val_loss']
    else:
        acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(nb_epoch), acc, label='Train')
    plt.plot(range(nb_epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(nb_epoch), loss, label='Train')
    plt.plot(range(nb_epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(nb_epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    if ifsaving:
    	plt.savefig('figure/figure_epoch_'+str(epochs_init)+'+'+str(epochs_training)+'.eps', format='eps', dpi=1000)
    plt.show()
