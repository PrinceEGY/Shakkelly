import keras
import keras.ops as ops


def masked_loss(labels, preds):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=None
    )
    loss = loss_fn(labels, preds)

    mask = labels != 0
    mask = ops.cast(mask, loss.dtype)

    loss = loss * mask
    loss = ops.sum(loss) / ops.sum(mask)
    return loss
