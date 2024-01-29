import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Load training and evaluation logs
df_160m_train = pd.read_csv('ttl-160m-train-loss-logs.csv')
df_460m_train = pd.read_csv('ttl-460m-train-loss-logs.csv')
df_160m_test = pd.read_csv('ttl-160m-test-loss-logs.csv')
df_460m_test = pd.read_csv('ttl-460m-test-loss-logs.csv')
df_460m_benchmark = pd.read_csv('460m-arc.csv')

# Smooth the training loss using a rolling average
window_size = 50_000  
df_160m_train['smoothed_loss'] = df_160m_train['loss'].rolling(window=window_size, min_periods=1, center=True).mean()
df_460m_train['smoothed_loss'] = df_460m_train['loss'].rolling(window=window_size, min_periods=1, center=True).mean()

# Get the training steps and losses as NumPy arrays
steps_160m = df_160m_train.step.values
steps_460m = df_460m_train.step.values
training_loss_160m = df_160m_train.smoothed_loss.values
training_loss_460m = df_460m_train.smoothed_loss.values

# Create the training loss plot
plt.figure(figsize=(15, 8))

# Add training loss lines
plt.plot(steps_160m, training_loss_160m, label='TTL-160m', color=(0.0, 0.8, 0.4), linestyle='-', linewidth=2.5)
plt.plot(steps_460m, training_loss_460m, label='TTL-460m', color=(0.5, 0.0, 0.5), linestyle='-', linewidth=2.5)

# Add markers at the final points
plt.scatter(steps_160m[-1], training_loss_160m[-1], color=(0.0, 0.8, 0.4), marker='x', linewidth=2.5, s=200)
plt.scatter(steps_460m[-1], training_loss_460m[-1], color=(0.5, 0.0, 0.5), marker='x', linewidth=2.5, s=200)

# Add horizontal dotted line at the final loss values
plt.axhline(y=training_loss_160m[-1], color=(0.0, 0.8, 0.4), linestyle='--')
plt.axhline(y=training_loss_460m[-1], color=(0.5, 0.0, 0.5), linestyle='--')

# Annotate the final points with text
plt.text(steps_160m[-1], training_loss_160m[-1], f'{training_loss_160m[-1]:.2f}', ha='right', va='bottom', fontsize=22, color="black")
plt.text(steps_460m[-1], training_loss_460m[-1], f'{training_loss_460m[-1]:.2f}', ha='right', va='bottom', fontsize=22, color="black")

# Customize the plot and save it as PNG and SVG
plt.xlabel('Processed Tokens (billions)', fontsize=26, fontname='Calibri')
plt.ylabel('Training Loss', fontsize=26, fontname='Calibri')
plt.legend()
plt.grid(True)
custom_ticks = [200000, 400000, 600000, 800000, 1000000, 1200000]
plt.xticks(custom_ticks, labels=[str(x)[:1] + "." + str(x)[1] + "B" for x in [8192 * x for x in custom_ticks]], fontsize=22, fontname='Calibri')
plt.yticks(fontsize=14, fontname='Calibri')
plt.legend(loc='upper right', fontsize='26')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.savefig("training-loss-plot.png", dpi=300, transparent=True)
plt.savefig("training-loss-plot.svg", format="svg", transparent=True)
plt.show()

# Get the evaluation steps and perplexity as NumPy arrays
stepes_eval_160m = df_160m_test['Steps'].values
stepes_eval_460m = df_460m_test['Steps'].values
perplexity_160m = df_160m_test['Perplexity'].values
perplexity_460m = df_460m_test['Perplexity'].values

# Create the perplexity plot
plt.figure(figsize=(15, 8))

# Add perplexity lines
plt.plot(stepes_eval_160m, perplexity_160m, label='TTL-160m', color=(0.0, 0.8, 0.4), linestyle='-', linewidth=2.5)
plt.plot(stepes_eval_460m, perplexity_460m, label='TTL-460m', color=(0.5, 0.0, 0.5), linestyle='-', linewidth=2.5)

# Add markers at the final points
plt.scatter(stepes_eval_160m[-1], perplexity_160m[-1], color=(0.0, 0.8, 0.4), marker='x', linewidth=2.5, s=200)
plt.scatter(stepes_eval_460m[-1], perplexity_460m[-1], color=(0.5, 0.0, 0.5), marker='x', linewidth=2.5, s=200)

# Add horizontal dotted line at the final perplexity values
plt.axhline(y=perplexity_160m[-1], color=(0.0, 0.8, 0.4), linestyle='--')
plt.axhline(y=perplexity_460m[-1], color=(0.5, 0.0, 0.5), linestyle='--')

# Annotate the final points with text
plt.text(stepes_eval_160m[-1], perplexity_160m[-1], f'{perplexity_160m[-1]:.2f}', ha='right', va='bottom', fontsize=22, color="black")
plt.text(stepes_eval_460m[-1], perplexity_460m[-1], f'{perplexity_460m[-1]:.2f}', ha='right', va='bottom', fontsize=22, color="black")

# Customize the plot and save it as PNG and SVG
plt.xlabel('Processed Tokens (billions)', fontsize=26, fontname='Calibri')
plt.ylabel('Perplexity', fontsize=26, fontname='Calibri')
plt.legend()
plt.grid(True)
plt.xticks(custom_ticks, labels=[str(x)[:1] + "." + str(x)[1] + "B" for x in [8192 * x for x in custom_ticks]], fontsize=22, fontname='Calibri')
plt.yticks(fontsize=14, fontname='Calibri')
plt.legend(loc='upper right', fontsize='26')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.savefig("evaluation-perplexity-plot.png", dpi=300, transparent=True)
plt.savefig("evaluation-perplexity-plot.svg", format="svg", transparent=True)
plt.show()

# Get the training steps and accuracy values as NumPy arrays
steps_460m = df_460m_benchmark.step.values
arc_acc = df_460m_benchmark.arc.values

# Create the acc plot
plt.figure(figsize=(15, 8))

# Add accuracy lines
plt.plot(steps_460m, arc_acc, label='TTL-460m', color=(0.5, 0.0, 0.5), linestyle='-', linewidth=3.5)

# Add horizontal dotted line at the final perplexity values
plt.axhline(y=arc_acc.mean(), color="red", linestyle='--')

# Annotate the final points with text
plt.text(steps_460m[-4], arc_acc.mean(), f'{arc_acc.mean():.2f}', ha='right', va='bottom', fontsize=16, color="black")

# Customize the plot and save it as PNG and SVG
plt.xlabel('Processed Tokens (billions)', fontsize=18, fontname='Calibri')
plt.ylabel('Accuracy (%)', fontsize=18, fontname='Calibri')
plt.legend()
plt.grid(True)
plt.xticks(custom_ticks, labels=[str(x)[:1] + "." + str(x)[1] + "B" for x in [8192 * x for x in custom_ticks]], fontsize=14, fontname='Calibri')
plt.yticks(fontsize=14, fontname='Calibri')
plt.legend(loc='upper right', fontsize='18')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.savefig("ttl-460m-arc-plot.png", dpi=300, transparent=True)
plt.savefig("ttl-460m-arc-plot.svg", format="svg", transparent=True)
plt.show()