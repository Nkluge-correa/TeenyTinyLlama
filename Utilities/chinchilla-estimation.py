def calculate_loss(N, D):
    # Parameters from the [Chinchilla paper](https://arxiv.org/abs/2203.15556)
    A = 406.4
    B = 410.7
    E = 1.69
    alpha = 0.32
    beta = 0.28

    # Calculate the loss using the scaling law
    loss = (A / N**alpha) + (B / D**beta) + E
    return loss

N = 460_000_000  # Number of parameters
D = 9_300_000_000  # Number of tokens in the training dataset

resulting_loss = calculate_loss(N, D)

print(f"The calculated loss for:\nN = {N:,} parameters\nD = {D:,} tokens\nIs: {resulting_loss:.2f}")
print(f"Which is equivalent to a perplexity of {2**resulting_loss:.2f}")
print("Equates to {0:.2f} tokens per parameter".format(D / N))