import matplotlib.pyplot as plt
COLOR_LIST = ['b', 'g', 'r', 'k', 'y']


def loss_curve(data: dict, name, rate=1):
    plt.figure()
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签
    last_loss = []
    for i, label in enumerate(sorted(data.keys(),reverse=True)):
        loss = data[label]
        last_loss.append(str(round(loss[-1], 4)))
        x_train_loss = range(int((1 - rate) * len(loss)), len(loss))  # loss的数量，即x轴
        y_total_loss = loss[int((1 - rate) * len(loss)):]
        plt.plot(x_train_loss, y_total_loss, linewidth=1, color=COLOR_LIST[i], linestyle="solid", label=label)

    plt.legend()
    plt.title('Loss curve : ' + last_loss[0] + " = " + "+".join(last_loss[1:]))
    plt.savefig(f'out/loss_curve/{name}-{rate}.png')
    plt.close()
