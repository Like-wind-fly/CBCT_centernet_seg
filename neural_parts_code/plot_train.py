import matplotlib.pyplot as plt

# Parse the log file and extract the metrics
epochs = []
loss = []
dice_loss_identify = []
dice_loss_identify_down = []
loss_identify_ce = []
loss_identify_down_ce = []
optimizer = []

with open('/home/yisiinfo/cyj/cj/segmentation_teeth_3d/neral-parts-cbct/demo/output/sga_net/stats.txt', 'r') as f:

    for line in f:
        for line in f:
            epoch, batch, loss_value, dice_loss_identify_value, dice_loss_identify_down_value, loss_identify_ce_value, loss_identify_down_ce_value, optimizer_value = line.strip().split(' ')

            epochs.append((epoch[0]))
            loss.append((loss_value)[0])
            dice_loss_identify.append((dice_loss_identify_value)[0])
            dice_loss_identify_down.append((dice_loss_identify_down_value)[0])
            loss_identify_ce.append((loss_identify_ce_value)[0])
            loss_identify_down_ce.append((loss_identify_down_ce_value)[0])
            optimizer.append((optimizer_value)[0])

 # Plot the metrics
plt.figure(figsize=(10, 10))

plt.subplot(2, 3, 1)
plt.plot(epochs, loss)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.subplot(2, 3, 2)
plt.plot(epochs, dice_loss_identify)
plt.xlabel('Epoch')
plt.ylabel('Dice Loss (Identify)')
plt.title('Dice Loss (Identify)')

plt.subplot(2, 3, 3)
plt.plot(epochs, dice_loss_identify_down)
plt.xlabel('Epoch')
plt.ylabel('Dice Loss (Identify Down)')
plt.title('Dice Loss (Identify Down)')

plt.subplot(2, 3, 4)
plt.plot(epochs, loss_identify_ce)
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss (Identify)')
plt.title('Cross Entropy Loss (Identify)')

plt.subplot(2, 3, 5)
plt.plot(epochs, loss_identify_down_ce)
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss (Identify Down)')
plt.title('Cross Entropy Loss (Identify Down)')
plt.subplot(2, 3, 6)
plt.plot(epochs, optimizer)
plt.xlabel('Epoch')
plt.ylabel('Optimizer')
plt.title('Optimizer')

plt.tight_layout()
plt.savefig('train_metrics.png')
