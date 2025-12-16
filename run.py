from src import train,evaluate_model
from src.evaluate import plot_lossG,read_losses_from_log,plot_Adv,plot_L1

if __name__ == "__main__":
    # train()
    evaluate_model()
    # dict = read_losses_from_log("outputs/logs/GAN_print.txt")
    # plot_lossG(lossG= dict["lossG"])
    # plot_Adv(Adv=dict['Adv'])
    # plot_L1(L1=dict['L1'])

