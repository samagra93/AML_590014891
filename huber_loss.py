def huber_loss(y, y_pred, delta=1.5):
    error = y - y_pred
    if abs(error) <= delta:
        return 0.5 * error ** 2
    else:
        return delta * (abs(error) - 0.5 * delta)

y_true = 10
y_pred = 13
loss = huber_loss(y_true, y_pred)
print("Huber Loss:", loss)