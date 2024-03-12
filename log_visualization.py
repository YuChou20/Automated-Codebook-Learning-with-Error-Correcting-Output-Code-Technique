import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def get_acc_array(filepath):
    file = open(filepath, 'r')
    lines = file.read().splitlines()
    file.close()
    acc_array = {}
    for i in range(100, 2001, 100):
        acc_array[i] = []
    ind = 0
    for line in lines:
        if not line or line.find('INFO:root:Epoch 199') == -1:
            continue
        # print(line)
        ind = ind % 2000 +100
        acc_ind = line.find('INFO:root:Epoch 199') + len('Top1 Test accuracy: ')
        acc_ind = line.find('Top1 Test accuracy: ') + len('Top1 Test accuracy: ')
        acc = round(float(line[acc_ind: acc_ind+6]),2)
        acc_array[ind].append(acc)
    acc_mean = []
    std = []
    for i in range(100, 2001, 100):
        acc_mean.append(round(np.average(acc_array[i]),2))
        std.append(round(np.std(acc_array[i]),2))
    return np.array(acc_mean), np.array(std)

def main():
    dataset = 'cifar10'
    index = [i for i in range(100,2001, 100)] 
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'tab:orange', 'tab:purple']
    i = 0
    for finetune in ['original_finetune', 'ecoc_decoder_finetune']:
        for method in ['baseline', 'proposed']:
            for outdim in ['100', '2048']:
                filepath = './runs/results/{0}_{1}_out{2}_{3}_None.log'.format(dataset, method, outdim, finetune)
                print(filepath)
                acc_mean, std = get_acc_array(filepath)
                print('Mean of acc:', acc_mean)
                print('Std of acc: ', std)
                print('Highest acc: {0} when epoch = {1}, std = {2}'.format(np.max(acc_mean), np.argmax(acc_mean)*100+100, std[np.argmax(acc_mean)]))
                # fig, ax = plt.subplots()
                x = index
                plt.plot(x, acc_mean, "-o", label='{0}-{1}-out{2}'.format(finetune, method, outdim))
                plt.fill_between(
                    x, acc_mean-std*5, acc_mean+std*5, color=colors[i], alpha=.05)
                plt.ylim([73,91])
                plt.xticks(index)
                i += 1
                plt.grid(True)
                # Show highest acc on the graph
                # plt.annotate(f'{np.max(acc_mean)}', (np.argmax(acc_mean, axis=0)*100+100, np.max(acc_mean)), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title('Baseline vs Proposed Via Different Outdim')
    plt.xlabel('Epoch')
    plt.ylabel('Top1 Acc')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

