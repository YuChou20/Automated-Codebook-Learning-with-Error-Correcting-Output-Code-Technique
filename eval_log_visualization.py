import matplotlib.pyplot as plt

def get_acc_array(filepath):
    
    file = open(filepath, 'r')
    lines = file.read().splitlines()
    file.close()
    acc1_array = []
    # print(lines)
    for line in lines:
        if not line:
            continue
        acc1_ind = line.find('Top1 Test accuracy: ') + len('Top1 Test accuracy: ')
        acc1 = round(float(line[acc1_ind: acc1_ind+6]),2)
        acc1_array.append(acc1)
    return acc1_array

def main():
    index = [i for i in range(1,201)] 
    filepath = './runs/{0}/eval.log'.format('cifar10-1000-lars-v5-1')
    acc = get_acc_array(filepath)
    filepath = './runs/{0}/eval.log'.format('cifar10-1000-lars-v5-baseline-1')
    acc2 = get_acc_array(filepath)

    print(acc)

    # Visualize loss history
    plt.plot(index, acc, 'r--')
    plt.plot(index, acc2, 'b-')
    plt.legend(['v5', 'v5-baseline'])
    plt.xlabel('Epoch')
    plt.ylabel('Top1 Acc')
    plt.show()
    # plt.plot(index, acc, color='b')
    # plt.xlabel('epoch') # 設定x軸標題
    # plt.xticks(index) # 設定x軸label以及垂直顯示
    # plt.locator_params(axis='x', nbins=10)
    # plt.title('Acc@1') # 設定圖表標題
    # plt.show()

if __name__ == "__main__":
    main()

