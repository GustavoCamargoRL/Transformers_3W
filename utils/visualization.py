import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as fp
import math


def result_visualization(loss_list: list,
                         correct_on_test: list,
                         correct_on_train: list,
                         test_interval: int,
                         d_model: int,
                         q: int,
                         v: int,
                         h: int,
                         N: int,
                         dropout: float,
                         DATA_LEN: int,
                         BATCH_SIZE: int,
                         time_cost: float,
                         EPOCH: int,
                         draw_key: int,
                         reslut_figure_path: str,
                         optimizer_name: str,
                         file_name: str,
                         LR: float,
                         pe: bool,
                         mask: bool):
    my_font = fp(fname=r"font/simsun.ttc")  # Set font path

    # Set plot style
    plt.style.use('seaborn-v0_8-pastel')

    fig = plt.figure()
    ax1 = fig.add_subplot(311)  # Top subplot
    ax2 = fig.add_subplot(313)  # Bottom subplot

    ax1.plot(loss_list, label="Loss")
    ax2.plot(correct_on_test, color='red', label='Accuracy on Test Dataset')
    ax2.plot(correct_on_train, color='blue', label='Accuracy on Train Dataset')

    # Axis labels and titles
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_xlabel(f'Epoch / {test_interval}')
    ax2.set_ylabel('Accuracy (%)')
    ax1.set_title('LOSS')
    ax2.set_title('ACCURACY')

    plt.legend(loc='best')

    # Add summary text to figure
    fig.text(x=0.13, y=0.4,
             s=f'Min loss: {min(loss_list)}    '
               f'Epoch at min loss: {math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}    '
               f'Final loss: {loss_list[-1]}\n'
               f'Max accuracy: Test {max(correct_on_test)}%  Train {max(correct_on_train)}%    '
               f'Epoch at max test accuracy: {(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}    '
               f'Final test accuracy: {correct_on_test[-1]}%\n'
               f'd_model={d_model}   q={q}   v={v}   h={h}   N={N}  dropout={dropout}\n'
               f'Total time: {round(time_cost, 2)} minutes',
             fontproperties=my_font)

    # Save figure (only if enough epochs have passed)
    if EPOCH >= draw_key:
        plt.savefig(
            f'{reslut_figure_path}/{file_name} {max(correct_on_test)}% {optimizer_name} '
            f'epoch={EPOCH} batch={BATCH_SIZE} lr={LR} pe={pe} mask={mask} '
            f'[{d_model},{q},{v},{h},{N},{dropout}].png')

    # Show figure
    plt.show()

    # Console outputs
    print('Accuracy list (test set):', correct_on_test)

    print(f'Min loss: {min(loss_list)}\r\n'
          f'Epoch at min loss: {math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}\r\n'
          f'Final loss: {loss_list[-1]}\r\n')

    print(f'Max accuracy: Test {max(correct_on_test)}%   Train {max(correct_on_train)}%\r\n'
          f'Epoch at max test accuracy: {(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}\r\n'
          f'Final test accuracy: {correct_on_test[-1]}%')

    print(f'Total training time: {round(time_cost, 2)} minutes')
